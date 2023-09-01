from ultralytics import YOLO
import torch
import time
import gc
if torch.cuda.is_available():
    print("Cuda available")
else:
    print("Shit?")


epochs = 100
patience = 100
image_size = 640
save = True
save_period = 1
cache = False
device = 0
workers = 8
project = 'training'
exist_ok = False
optimizer = 'Adam'
verbose = False
seed = 42
close_mosaic = 10
resume = False
lr0 = 0.01
lrf = 0.01
pretrained = False
batch_size = [4, 4, 4, 4, 4, 4]

model_names = ['yolov6n.yaml','yolov8n.yaml','yolov6s.yaml', 'yolov8s.yaml','yolov6m.yaml', 'yolov8m.yaml']


datasets = ['SKU-110K.yaml', 'SKU-110K-VS.yaml', 'supermarket_datasetV2.yaml']


# train models on original SKU-110K dataset, augmented SKU-110K-VS dataset and SupermarketV2 dataset
for dataset in datasets:
    for i, model_name in enumerate(model_names):
        model = YOLO(model_name)
        try:
            model.train(data=dataset, epochs=epochs,patience=patience,batch=batch_size[i], imgsz=image_size,save=save,save_period=save_period,cache=cache,device=device, 
                        workers=workers,project=project, name=model_name.replace('.yaml', '_')+dataset.replace('.yaml', ''),exist_ok=exist_ok,pretrained=pretrained,
                        optimizer=optimizer,verbose=verbose,seed=seed,close_mosaic=close_mosaic, lrf=lrf)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory, skipping to next model...")
        # release gpu memory
        time.sleep(15)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(15)

# train on supermarket_datasetv2 with pretrained SKU-110K and SKU-110K-VS weights
pretrained = True
datasets = ['SKU-110K.yaml', 'SKU-110K-VS.yaml']

for dataset in datasets:
    for i, model_name in enumerate(model_names):
        model = YOLO("training/" + model_name.replace('.yaml','_') + dataset.replace('.yaml','') + '/weights/best.pt')
        try:
            model.train(data='supermarket_datasetV2.yaml', epochs=epochs,patience=patience,batch=batch_size[i], imgsz=image_size,save=save,save_period=save_period,cache=cache,device=device, 
                        workers=workers,project=project, name=model_name.replace('.yaml', '_')+'pretrainedOn' + dataset.replace('.yaml', '') + '_supermarket_datasetV2.yaml',exist_ok=exist_ok,pretrained=pretrained,
                        optimizer=optimizer,verbose=verbose,seed=seed,close_mosaic=close_mosaic, lrf=lrf)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory, skipping to next model...")
        # release gpu memory
        time.sleep(15)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(15)


# Transfer learning
frozen_layers = [3, 5, 7, 9]

for fl in frozen_layers:
    
    def freeze_layer(trainer):
        num_freeze = fl
        model = trainer.model
        print(f"Freezing {num_freeze} layers")
        freeze = [f'model.{x}.' for x in range(num_freeze)] # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True # train all layers
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False
                print(f"{num_freeze} layers are frozen.")

    pretrained = True
    datasets = ['SKU-110K.yaml', 'SKU-110K-VS.yaml']

    for dataset in datasets:
        for i, model_name in enumerate(model_names):
            model = YOLO("training/" + model_name.replace('.yaml','_') + dataset.replace('.yaml','') + '/weights/best.pt')
            model.add_callback("on_train_start", freeze_layer)
            try:
                model.train(data='supermarket_datasetV2.yaml', epochs=epochs,patience=patience,batch=batch_size[i], imgsz=image_size,save=save,save_period=save_period,cache=cache,device=device, 
                            workers=workers,project=project, name=model_name.replace('.yaml', '_')+'pretrainedOn' + dataset.replace('.yaml', '') + "_transfer_freeze" +str(fl) + '_supermarket_datasetV2.yaml',exist_ok=exist_ok,pretrained=pretrained,
                            optimizer=optimizer,verbose=verbose,seed=seed,close_mosaic=close_mosaic, lrf=lrf)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("CUDA out of memory, skipping to next model...")
            # release gpu memory
            time.sleep(15)
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(15)