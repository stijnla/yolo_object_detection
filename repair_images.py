import cv2

from os import listdir
from os.path import isfile, join

testset = '../datasets/SKU-110K-VS/test.txt'
trainset = '../datasets/SKU-110K-VS/train.txt'
valset = '../datasets/SKU-110K-VS/val.txt'

new_path = '../datasets/SKU-110K-VS/images/'
onlyfiles = [f for f in listdir(new_path) if isfile(join(new_path, f))]

with open(testset, 'r') as f:
    testlines = f.readlines()
with open(trainset, 'r') as f:
    trainlines = f.readlines()
with open(valset, 'r') as f:
    vallines = f.readlines()

lines = testlines + trainlines + vallines
newlines = []
for l in lines:
    l = l.replace('\n', '')
    l = l[9::]
    newlines.append(l)

for im in onlyfiles:
    if im not in newlines:
        print(im)

print(len(newlines))
print(len(onlyfiles))