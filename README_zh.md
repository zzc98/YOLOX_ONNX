# YOLOX 进行车辆检测

## 项目简介

使用yolox进行车辆检测。

数据集：[SODA-10M](https://soda-2d.github.io/)

## 环境依赖

- onnxruntime==1.14.1
- opencv-python==4.7.0.72
- torchvision== 0.14.1+cu117
- Pillow==9.4.0

## 训练

基于[mmdet](https://github.com/open-mmlab/mmdetection)平台，训练轻量级模型yolox-tiny。训练配置见[config](SSLAD.py)。

## 推理

模型训练完毕后，导出为onnx格式，无需在代码中显示定义模型结构，可直接输入图片进行车辆行人检测。详见[demo](yolox_onnx.ipynb)。

