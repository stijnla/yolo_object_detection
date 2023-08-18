import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading the CSV file
csvFilen = pd.read_csv('SKU-110K_experiments/experiment_nano_noAugment/results.csv')
csvFiles = pd.read_csv('SKU-110K_experiments/experiment_small_noAugment/results.csv')

# displaying the contents of the CSV file
print(csvFilen.keys())

epochs = csvFilen['                  epoch'].to_numpy()
train_box_lossn = csvFilen['         train/box_loss'].to_numpy()
val_box_lossn = csvFilen['           val/box_loss'].to_numpy()
train_cls_lossn = csvFilen['         train/cls_loss'].to_numpy()
val_cls_lossn = csvFilen['           val/cls_loss'].to_numpy()
train_dfl_lossn = csvFilen['         train/dfl_loss'].to_numpy()
val_dfl_lossn = csvFilen['           val/dfl_loss'].to_numpy()
train_box_losss = csvFiles['         train/box_loss'].to_numpy()
val_box_losss = csvFiles['           val/box_loss'].to_numpy()
train_cls_losss = csvFiles['         train/cls_loss'].to_numpy()
val_cls_losss = csvFiles['           val/cls_loss'].to_numpy()
train_dfl_losss = csvFiles['         train/dfl_loss'].to_numpy()
val_dfl_losss = csvFiles['           val/dfl_loss'].to_numpy()

# plot lines
plt.plot(epochs, train_box_lossn, label = "train_box_loss_nano")
plt.plot(epochs, val_box_lossn, label = "val_box_loss_nano")
plt.plot(epochs, train_box_losss, label = "train_box_loss_small")
plt.plot(epochs, val_box_losss, label = "val_box_loss_small")
plt.legend()
plt.show()

plt.plot(epochs, train_cls_lossn, label = "train_cls_loss_nano")
plt.plot(epochs, val_cls_lossn, label = "val_cls_loss_nano")
plt.plot(epochs, train_cls_losss, label = "train_cls_loss_small")
plt.plot(epochs, val_cls_losss, label = "val_cls_loss_small")
plt.legend()
plt.show()

plt.plot(epochs, train_dfl_lossn, label = "train_dfl_loss_nano")
plt.plot(epochs, val_dfl_lossn, label = "val_dfl_loss_nano")
plt.plot(epochs, train_dfl_losss, label = "train_dfl_loss_small")
plt.plot(epochs, val_dfl_losss, label = "val_dfl_loss_small")
plt.legend()
plt.show()