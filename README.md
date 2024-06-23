# Person Re-Identification in a Video Sequence

## Overview
This repository contains the implementation of our project on person re-identification in video sequences using deep learning techniques. We leverage OSNet for feature extraction and YOLOv8 for object detection to build a robust application that identifies and tracks individuals in video sequences.

## Paper
- **Authors**: Zhiyuan Li, Jiayang Wang
- **Institution**: Stanford University
- **Abstract**: In this project, we train and refine the OSNet model using transfer learning on Market1501 and DukeMTMC-reID datasets. We then build an application to identify and track persons of interest in video sequences.

## Models and Dependencies

### YOLOv8
We utilize the YOLOv8 model for object detection due to its high performance in real-time applications.
- GitHub: [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)

### OSNet
For person re-identification, we use the OSNet model, known for its omni-scale feature learning capabilities.
- GitHub: [OSNet by KaiyangZhou](https://github.com/KaiyangZhou/deep-person-reid)
