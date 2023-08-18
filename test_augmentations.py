import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

#random.seed(42)

image = np.array(plt.imread('koffie_schap.jpg'))


images=[]

# Spatial
transform = A.Rotate(p=1, limit=10)
transform = A.Perspective(p=1)
transform = A.HorizontalFlip(p=1)

# Blur
transform = A.MotionBlur(p=1, blur_limit=(3,37))

# Visual
transform = A.RandomToneCurve(p=1)
transform = A.Sharpen(p=1)
transform = A.RandomGamma(p=1)
transform = A.RandomBrightnessContrast(p=1)
transform = A.HueSaturationValue(p=1, hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=20)
transform = A.CLAHE(p=1)

# Noise
transform = A.ISONoise(p=1)
transform = A.MultiplicativeNoise(p=1)
transform = A.GaussNoise(p=1, per_channel=False, var_limit=(10, 255))
transform = A.GaussNoise(p=1, per_channel=True, var_limit=(10, 255))


for i in range(100):
    transformed = transform(image=image)
    images.append(transformed['image'])


for img in images:
    new_image = np.hstack((image, img))
    cv2.imshow("image", cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(100)    