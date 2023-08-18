from ultralytics import YOLO

import cv2
# Load YOLO model trained for 10 epochs on SKU-110K-VS dataset
model = YOLO('runs/detect/yolo_large_supermarket_batchsz1/weights/last.pt')

# Perform object detection on an image using the model
results = model.predict(source='koffie_schap.jpg', show=True, save=False)

# Get bounding boxes defined as (x, y, widht, height)
bounding_boxes_normalized = results[0].boxes.xywhn
bounding_boxes = results[0].boxes.xywh
# Get confidence values (scores) of each bounding box
scores = results[0].boxes.conf

print(bounding_boxes[0])
image = cv2.imread('koffie_schap.jpg')
print(bounding_boxes_normalized[0][0]*image.shape[1])
print(bounding_boxes_normalized[0][1]*image.shape[0])
print(bounding_boxes_normalized[0][2]*image.shape[1])
print(bounding_boxes_normalized[0][3]*image.shape[0])
