# YOLOX for Vehicle Detection

## Project Introduction

vehicle detection using yolox.

Dataset: [SODA-10M](https://soda-2d.github.io/)

## Dependencies
- onnxruntime==1.14.1
- opencv-python==4.7.0.72
- torchvision== 0.14.1+cu117
- Pillow==9.4.0

## Training

Train lightweight model yolox-tiny based on [mmdet](https://github.com/open-mmlab/mmdetection) platform. Training configuration is in [config](SSLAD.py).

## Inference

The model is exported to onnx format after training, which does not need to display model structure in the code, and can directly input images for vehicle and pedestrian detection. See [demo](yolox_onnx.ipynb) for details.
