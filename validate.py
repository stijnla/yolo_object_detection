from ultralytics import YOLO

# Load YOLO model trained for 10 epochs on SKU-110K-VS dataset
model = YOLO('training/yolov6n_SKU-110K-VS/weights/best.pt')

# Validate model on test split
metrics = model.val(data='SKU-110K.yaml', split='test', device=0)

print(metrics.box.map) # map 50-95
print(metrics.box.map75)
print(metrics.box.map50)
print(metrics.box.maps)
metrics = model.val(data='SKU-110K-VS.yaml', split='test', device=0)
print(metrics.box.map) # map 50-95
print(metrics.box.map75)
print(metrics.box.map50)
print(metrics.box.maps)
metrics = model.val(data='supermarket_datasetV2_nolabel.yaml', split='test', device=0)
print(metrics.box.map) # map 50-95
print(metrics.box.map75)
print(metrics.box.map50)
print(metrics.box.maps)