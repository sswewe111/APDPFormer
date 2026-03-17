# APDRFormer: Asymmetric Perception Decoupling and Recovery Transformer for Multimodal 3D Object Detection

## Abstract

## Main Results

#### 3D object detection results on nuScenes dataset.

| Method                   | Modality | mAP (val) | NDS (val) | mAP (test) | NDS (test) |
| ------------------------ | -------- | --------- | --------- | ---------- | ---------- |
| TransFusion-L (Baseline) | L        | 65.1      | 70.1      | 65.5       | 70.2       |
| TransFusion-LC           | L+C      | 67.5      | 71.3      | 68.9       | 71.7       |
| BEVFusion                | L+C      | 68.5      | 71.4      | 70.2       | 72.9       |
| APDRFormer (Ours)        | L+C      | 72.9      | 74.3      | 73.2       | 75.4       |


## Use APDPFormer

### Installation

This project is based on torch 1.10.1, mmdet 2.14.0, mmcv 1.4.0 and mmdet3d 0.16.0. Please install mmdet3d following [getting_started.md](docs/getting_started.md). 

### Dataset Preparation
Please refer to [data_preparation.md](docs/data_preparation.md) to prepare the nuScenes dataset.
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0
```

### Training and Evaluation
Pre-trained weight file [base.pth](https://pan.baidu.com/s/14S3jrdc83tsfRP8KLeoIoA?pwd=1234)
Start the training and evluation by running:

```
bash tools/run-nus.sh extra-tag
```
To obtain detection results using the pretrained model, run the following command:
```
bash tools/dist_test.sh configs/apdpformer_v1.py path_to_ckpt_directory 1 --eval bbox
```

