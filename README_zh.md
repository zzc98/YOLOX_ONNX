# YOLOX 进行车辆检测

## 项目简介

YOLOX是anchor-free版本的YOLO，设计更简单，性能更强。由于参数量小，可以方便部署到移动端进行实时目标检测。

本项目使用yolox进行车辆检测。

## 环境依赖

- numpy
- onnxruntime
- opencv-python
- Pillow

## 数据集

项目使用的数据集为[SODA-10M](https://soda-2d.github.io/)。包含六个类别：
- 行人
- 骑自行车的人
- 汽车
- 卡车
- 有轨电车
- 三轮车

## 训练

基于[mmdet](https://github.com/open-mmlab/mmdetection)平台，训练轻量级模型yolox-tiny。仅使用训练集进行微调.

## 推理

模型训练完毕后，导出为onnx格式，无需在代码中显示定义模型结构，可直接输入图片进行车辆行人检测。详见[demo](yolox_onnx.ipynb)。

## 致谢

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
