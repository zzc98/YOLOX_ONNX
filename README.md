# YOLOX for Vehicle Detection

## Project Introduction


YOLOX is the anchor-free version of YOLO with simpler design and better performance. Due to the small number of parameters, it can be easily deployed to mobile for real-time object detection.

This project uses yolox for vehicle detection.


## Dependencies

- numpy
- onnxruntime
- opencv-python
- Pillow

## Dataset

The dataset used in the project is [SODA-10M](https://soda-2d.github.io/). Six categories are included:
- Pedestrian
- Cyclist
- Car
- Truck
- Tram
- Tricycle

## Training

Train lightweight model yolox-tiny based on [mmdet](https://github.com/open-mmlab/mmdetection) platform. Only the training set is used for fine-tuning.

## Inference

The model is exported to onnx format after training, which does not need to display model structure in the code, and can directly input images for vehicle and pedestrian detection. See [demo](yolox_onnx.ipynb) for details.

## Acknowledgement

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
