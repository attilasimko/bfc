import sys
import scipy
from tensorflow.keras import optimizers, metrics
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from scipy.stats import norm
import matplotlib.mlab as mlab
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sys
random.seed(2019)
# sys.path.append("C:/Users/attil/Documents/DRS/drs/")
# from MLTK.data import BiasGenerator


"""
Created: 2019. 03. 28.
Testing script for CompNet paper figure describing the network using
the training data. Show each stage with figures.

Stage 1:

Stage 2:

Stage 3:

"""

# model = load_model('Experiments/CompNets/000/12.h5')
model = tensorflow.keras.models.load_model('C:/Users/attil/Downloads/out/10.h5', compile=False)
dataset = '1TA40.npz'

with np.load(dataset) as npzfile:
    im = npzfile['image'][50, :, :]
    data = npzfile['data'][50, :, :]
    tissue = npzfile['tissue'][50, :, :]

BFG = BiasGenerator(size=(256, 256), batch_size=1)

bias = BFG.GetFields()
bias = bias[:, :, 0]
np.savez('data', image = im, data = data, bias = bias)
img2 = im

img1 = im * bias
img1 = img1 / np.max(img1)

plt.figure(1, figsize=(20, 5))
plt.subplot(2, 5, 1)
plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.subplot(2, 5, 6)
plt.imshow(bias, cmap='gray')
plt.axis('off')

plt.subplot(2, 5, 2)
plt.imshow(img1, cmap='gray')
plt.axis('off')


img1 = np.expand_dims(np.expand_dims(img1, axis=0), axis=4)
img2 = np.expand_dims(np.expand_dims(img2, axis=0), axis=4)

bias1 = model.predict(img1)
bias2 = model.predict(img2)
bias1 = bias1[0, :, :, 0]
bias2 = bias2[0, :, :, 0]
bias_pred = np.divide(bias1, bias2)
bias_pred = bias_pred / np.mean(bias_pred)

img1 = np.divide(img1[0, :, :, 0], bias1)
img2 = np.divide(img2[0, :, :, 0], bias2)
img1 = img1 / np.max(img1)
img2 = img2 / np.max(img2)

img1 = np.expand_dims(np.expand_dims(img1, axis=0), axis=4)
img2 = np.expand_dims(np.expand_dims(img2, axis=0), axis=4)

reg1 = model.predict(img1)
reg2 = model.predict(img2)
reg1 = reg1[0, :, :, 0]
reg2 = reg2[0, :, :, 0]

vmin1, vmax1 = np.min(bias1), np.max(bias1)
vmin2, vmax2 = np.min(bias2), np.max(bias2)
vmin3, vmax3 = np.min(bias_pred), np.max(bias_pred)
vmin4, vmax4 = np.min(reg1), np.max(reg1)
vmin5, vmax5 = np.min(reg2), np.max(reg2)
vmin = np.min([vmin1, vmin2, vmin3, vmin4, vmin5])
vmax = np.max([vmax1, vmax2, vmax3, vmax4, vmax5])

plt.subplot(2, 5, 3)
plt.imshow(bias1, cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(2, 5, 8)
plt.imshow(bias2, cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')

plt.subplot(2, 5, 4)
plt.imshow(reg1, cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(2, 5, 9)
plt.imshow(reg2, cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(bias_pred, cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.show()
