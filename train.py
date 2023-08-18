from ultralytics import YOLO
import torch
import time
if torch.cuda.is_available():
    print("Cuda available")
else:
    print("Shit?")


epochs = 1
patience = 100
batch_size = 4
image_size = 640
save = True
save_period = 1
cache = False
device = 0
workers = 8
project = 'training'
exist_ok = False
pretrained = False
optimizer = 'Adam'
verbose = True
seed = 42
close_mosaic = 10
resume = False
lr0 = 0.01
lrf = 0.01

datasets = ['SKU-110K.yaml', 'SKU-110K-VS.yaml']

model_names = ['yolov8n.yaml','yolov8s.yaml','yolov8m.yaml', 'yolov8x.yaml','yolov6n.yaml', 'yolov6s.yaml', 'yolov6m.yaml', 'yolov6l.yaml']

# train models on original SKU-110K dataset and augmented SKU-110K-VS dataset
for dataset in datasets:
    for model_name in model_names:
        model = YOLO(model_name)
        try:
            model.train(data=dataset, epochs=epochs,patience=patience,batch=batch_size, imgsz=image_size,save=save,save_period=save_period,cache=cache,device=device, workers=workers,
                        project=project, name=model_name.replace('.pt', '')+'_NoAugmentation',exist_ok=exist_ok,pretrained=pretrained,optimizer=optimizer,verbose=verbose,seed=seed,close_mosaic=close_mosaic, lrf=lrf)
        except:
            pass
        # release gpu memory
        time.sleep(15)
        torch.cuda.empty_cache()
        time.sleep(15)

