from ultralytics import YOLO

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

import torch
from torch.nn import functional as F
from torchreid.utils import FeatureExtractor

model = YOLO("yolov8n.pt")

# extractor = FeatureExtractor(model_name='osnet_x1_0', model_path='log/osnet_x1_0_dukemtmcreid_softmax_cosinelr_new_transfer_35_50/model/model.pth.tar-50', device='cuda')

extractor = FeatureExtractor(model_name='osnet_x1_0', model_path='log/osnet_x1_0_market1501_softmax_cosinelr/model/model.pth.tar-250', device='cuda')

video_path = 'TestVideo.avi'
cap = cv2.VideoCapture(video_path)

output_directory = 'cropped_images'
os.makedirs(output_directory, exist_ok=True)

# query_img = ['cropped_images/cropped_person_1040_1.jpg']
# query_img = ['cropped_images/cropped_person_669_0.jpg']
# query_img = ['cropped_images/cropped_person_1430_3.jpg']
# query_img = ['cropped_images/cropped_person_1102_4.jpg']
query_img = ['cropped_images/cropped_person_1278_1.jpg']
query_feature = extractor(query_img)
query_input = F.normalize(query_feature, p=2, dim=1)

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

  if len(person_indices) == 0:
    continue

  img_list = []
  input_list = []

  for idx in person_indices:
    x1, y1, x2, y2 = coordinates[idx]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    print(x1, y1, x2, y2)

    cropped_rgb_array = frame_rgb[y1:y2, x1:x2]
    cropped_image = Image.fromarray(cropped_rgb_array)

    image = cropped_image.resize((16, 64), Image.Resampling.LANCZOS)
    output_path = os.path.join(output_directory, f'tmp.jpg')
    image.save(output_path)
    img_list.append(image)
    feature = extractor(['cropped_images/tmp.jpg'])
    img_input = F.normalize(feature, p=2, dim=1)
    input_list.append(img_input)

  combined_tensor = torch.cat(input_list).view(len(input_list), -1)
  dist = 1 - torch.mm(query_input, combined_tensor.t())

  min_val, min_index = torch.min(dist, dim=1)
  print(dist)
 
  if min_val.item() < 0.20:
    # query_input = input_list[min_index.item()]
    out = os.path.join('resulting_persons', f'person_{frame_count}_{min_index.item()}_{min_val.item()}.jpg')
    img_list[min_index.item()].save(out)


