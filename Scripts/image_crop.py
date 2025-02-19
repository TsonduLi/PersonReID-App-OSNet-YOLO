from ultralytics import YOLO

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

model = YOLO("yolov8n.pt")

video_path = 'TestVideo.avi'
cap = cv2.VideoCapture(video_path)

output_directory = 'cropped_images'
os.makedirs(output_directory, exist_ok=True)

frame_count = 0

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  frame_count += 1

  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = model(frame_rgb)

  classes = results[0].boxes.cls
  coordinates = results[0].boxes.xyxy

  person_indices = (classes == 0).nonzero(as_tuple=True)[0]

  for idx in person_indices:
    x1, y1, x2, y2 = coordinates[idx]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    print(x1, y1, x2, y2)

    cropped_rgb_array = frame_rgb[y1:y2, x1:x2]
    cropped_image = Image.fromarray(cropped_rgb_array)

    image = cropped_image.resize((16, 64), Image.Resampling.LANCZOS)
    output_path = os.path.join(output_directory, f'cropped_person_{frame_count}_{idx}.jpg')
    image.save(output_path)


