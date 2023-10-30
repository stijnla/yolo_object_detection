import random

from os import listdir
from os.path import isfile, join

# randomly split data in train, val, and test splits
mypath = '../datasets/NAME_OF_YOUR_DATASET/images'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

trainset = []
valset = []
testset = []

for image_file in onlyfiles:
    r = random.randint(0,9)
    if r >= 7 and r < 9:
        valset.append(image_file)
    elif r == 9:
        testset.append(image_file)
    else:
        trainset.append(image_file)

with open('train.txt', 'w') as f:
    for image_file in trainset:
        f.write('./images/' + image_file + '\n')

with open('test.txt', 'w') as f:
    for image_file in testset:
        f.write('./images/' + image_file + '\n')

with open('val.txt', 'w') as f:
    for image_file in valset:
        f.write('./images/' + image_file + '\n')