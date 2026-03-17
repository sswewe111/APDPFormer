import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.utils.fusionmodule import LRF


class SFA(nn.Module):
    
    def __init__(self, in_channels=128, out_channels=128, num_classes=10, q=5, hidden_channels=64):
        super(SFA, self).__init__()
        assert q % 2 == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.q = q
        self.q2 = q * q
        self.max_offset = 5

        feat_in_ch = in_channels + self.q2 + in_channels + self.q2
        self.flow_net = nn.Sequential(
            nn.Conv2d(feat_in_ch, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=3, padding=1)  # 4 => (dx,dy) for P and I
        )

        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.fusion_layer = LRF([in_channels, in_channels], out_channels)

    def forward(self, pts_bev,img_bev,heatmap_pts,heatmap_img):
        B, C, H, W = pts_bev.shape       
        device = pts_bev.device
        dtype = pts_bev.dtype

        heatmap_pts=heatmap_pts.permute(0, 1, 3, 2).contiguous()
        heatmap_img=heatmap_img.permute(0, 1, 3, 2).contiguous()

        hp = heatmap_pts.detach().sigmoid()
        hi = heatmap_img.detach().sigmoid()

        cp = self._cost_volume(hp, hi)  
        ci = self._cost_volume(hi, hp)  

        concat = torch.cat([pts_bev, cp, img_bev, ci], dim=1)  # B x (2C+2*q2) x H x W
        flow_out = self.flow_net(concat)  # B x 4 x H x W
        flow_out = torch.tanh(flow_out) * self.max_offset
        delta_p = flow_out[:, 0:2, :, :]  
        delta_i = flow_out[:, 2:4, :, :]  

        
        base_grid = self._make_base_grid(B, H, W, device=device, dtype=dtype)
        
        delta_p_xy = delta_p.permute(0, 2, 3, 1)  # B x H x W x 2
        delta_i_xy = delta_i.permute(0, 2, 3, 1)  # B x H x W x 2
        src_p = base_grid + delta_p_xy  
        src_i = base_grid + delta_i_xy
        
        grid_p = self._pixel_to_grid_sample(src_p, H, W)  # B x H x W x 2
        grid_i = self._pixel_to_grid_sample(src_i, H, W)
        
        aligned_p = F.grid_sample(pts_bev, grid_p, mode='bilinear', padding_mode='border', align_corners=True)
        aligned_i = F.grid_sample(img_bev, grid_i, mode='bilinear', padding_mode='border', align_corners=True)
        
        #fused = self.fusion_conv(torch.cat([aligned_p, aligned_i], dim=1))
        fused=self.fusion_layer([aligned_i,aligned_p])

        return fused


    def _cost_volume(self, H_a, H_b):
        B, Nc, H, W = H_b.shape
        pad = self.q // 2
       
        H_a = F.normalize(H_a, p=2, dim=1)
        H_b = F.normalize(H_b, p=2, dim=1)
     
        patches = F.unfold(H_b, kernel_size=self.q, padding=pad, stride=1)  # B x (Nc*q2) x (H*W)
        patches = patches.reshape(B, Nc, self.q2, H * W)  # B, Nc, q2, HW

        ha = H_a.reshape(B, Nc, 1, H * W)  # B, Nc, 1, HW
        cost = (ha * patches).sum(dim=1)  # B x q2 x HW

        cost = cost.reshape(B, self.q2, H, W)
        return cost

    def _make_base_grid(self, B, H, W, device, dtype):
       
        xs = torch.arange(0, W, device=device, dtype=dtype)
        ys = torch.arange(0, H, device=device, dtype=dtype)
        ys, xs = torch.meshgrid(ys, xs)  # H x W
        grid = torch.stack((xs, ys), dim=-1)  # H x W x 2
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # B x H x W x 2
        return grid  

    def _pixel_to_grid_sample(self, px, H, W):
        
        x = px[..., 0]
        y = px[..., 1]
        x_norm = 2.0 * (x / (W - 1)) - 1.0
        y_norm = 2.0 * (y / (H - 1)) - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1)
        return grid