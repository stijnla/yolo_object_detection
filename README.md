# YOLO trainer
Implements scripts used to train and validate yolo models

## Requirements
 - pytorch <= 2.0.1
 - ultralytics <= 8.0.157

## Prepare dataset
To train a model properly, you need a dataset that represents your application
Make sure the dataset is split into three sets: 
 - train
 - val
 - test
These splits are preferably split equally, so each split contains the equal ratios of classes and scenarios
Splitting the dataset in three sets randomly can be done by running ***split_data.py***

Furthermore, to enable the ultralytics library to find the dataset, a ***.yaml*** file must be created
This file contains five keywords:
 - path
 - train
 - val
 - test
 - names

**path** contains the path to the dataset
The **train**, **val**, and **test** keywords point to the .txt files that contain the split information
**names** contains a list that contains all names of the classes (eg. 0=person, 1=car, ...)

## Train model
The ***train.py*** script runs the training program. Make sure to set the **dataset** parameter to the just created ***.yaml*** file. The ***pretrained_weights*** parameter determines what model is loaded:
 - nano (yolov8n.pt)
 - small (yolov8s.pt)
 - medium (yolov8m.pt)
 - large (yolov8l.pt)
 - extra large (yolov8x.pt)
 - pretrained supermarket model (SKU-110K_pretrained_yolov6_medium.pt)

