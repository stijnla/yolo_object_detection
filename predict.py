from ultralytics import YOLO

# Load YOLO model trained for 10 epochs on SKU-110K-VS dataset
model = YOLO('training/yolov6n_supermarket_datasetV2/weights/best.pt')

# Perform object detection on an image using the model
results = model.predict(source='rgb_easy.mp4', show=True, save=False, device=0)

# Get bounding boxes defined as (x, y, widht, height)
bounding_boxes_normalized = results[0].boxes.xywhn
bounding_boxes = results[0].boxes.xywh
# Get confidence values (scores) of each bounding box
scores = results[0].boxes.conf


