# Person Re-Identification in a Video Sequence

## Overview
This repository contains the implementation of our project on person re-identification in video sequences using deep learning techniques. We leverage OSNet for feature extraction and YOLOv8 for object detection to build a robust application that identifies and tracks individuals in video sequences.

## Features
In this project, we:
- **Train and Refine OSNet**: We train OSNet on the Market1501 dataset and refine it using transfer learning on the DukeMTMC-reID dataset.
- **Two-Step Transfer Learning**: We modify the configuration files of OSNet to implement a two-step transfer learning process. The new configuration files can be found in the `configs` folder.
- **Object Detection with YOLOv8**: We integrate YOLOv8 for real-time person detection in video sequences.
- **Application Development**: We build an application that takes a video sequence and a query image as input and outputs all occurrences of the queried person in the video.

## Models and Dependencies

### YOLOv8
We utilize the YOLOv8 model for object detection due to its high performance in real-time applications.
- GitHub: [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)

### OSNet
For person re-identification, we use the OSNet model, known for its omni-scale feature learning capabilities.
- GitHub: [OSNet by KaiyangZhou](https://github.com/KaiyangZhou/deep-person-reid)

## Authors
- Zhiyuan Li - Stanford University
- Jiayang Wang - Stanford University
