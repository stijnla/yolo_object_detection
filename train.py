from ultralytics import YOLO

epochs = 200
patience = 10
image_size = 640
save = True
save_period = 20
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
pretrained = True
batch_size = 8

model_name = 'new_model'
dataset = 
pretrained_weights = 


model = YOLO(pretrained_weights)
model.train(data=dataset, 
            epochs=epochs,
            patience=patience,
            batch=batch_size, 
            imgsz=image_size,
            save=save,
            save_period=save_period,
            cache=cache,
            device=device, 
            workers=workers,
            project=project, 
            name=model_name,
            exist_ok=exist_ok,
            pretrained=pretrained,
            optimizer=optimizer,
            verbose=verbose,
            seed=seed,
            close_mosaic=close_mosaic, 
            lrf=lrf)
