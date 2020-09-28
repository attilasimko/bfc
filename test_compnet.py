import sys
import scipy
import tensorflow
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


"""
Created: 2019. 03. 28.
Testing script for CompNet paper figure describing the network using
the training data. Show each stage with figures.

Stage 1:

Stage 2:

Stage 3:

"""

# model = load_model('Experiments/CompNets/000/12.h5')
model = load_model('C:/Users/attil/Downloads/out/100.h5')
dataset = '1TA40.npz'

with np.load(dataset) as npzfile:
    im = npzfile['image']
    data = npzfile['data']
    tissue = npzfile['tissue']

BFG = BiasGenerator(size=(256, 256), batch_size=1)

inputImage, inputBias = val_gen.__getitem__(0)
for fileID in range(20):
    img2 = inputImage[0][0, int(fileID*5), :, :, 0]
    true_bias = inputBias[0][0, int(fileID*5), :, :, 0]

    bias = BFG.GetFields()
    bias = bias[:, :, 0]
    img1 = img2 * bias
    img1 = img1 / np.max(img1)
    #bias = inputBias[0][fileID, :, :, 0]
    #bias = cv2.resize(bias, dsize=(256, 256),
    #                  interpolation=cv2.INTER_LANCZOS4)
    #bias = bias / np.mean(bias)

    plt.figure(1, figsize=(90, 10))
    plt.subplot(191)
    plt.imshow(img1*bias, cmap='gray')
    plt.axis('off')
    plt.subplot(192)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.subplot(193)
    plt.imshow(bias, cmap='gray')
    plt.axis('off')

    img1 = np.expand_dims(np.expand_dims(img1, axis=0), axis=4)
    img2 = np.expand_dims(np.expand_dims(img2, axis=0), axis=4)

    bias1 = model.predict(img1)
    bias2 = model.predict(img2)
    bias1 = bias1[0, :, :, 0]
    bias2 = bias2[0, :, :, 0]

    bias1 = bias1 / np.mean(bias1)
    bias2 = bias2 / np.mean(bias2)
    bias_pred = np.divide(bias1, bias2)
    bias_pred = bias_pred / np.mean(bias_pred)
    vmin1, vmax1 = np.min(bias1), np.max(bias1)
    vmin2, vmax2 = np.min(bias2), np.max(bias2)
    vmin = np.min([vmin1, vmin2])
    vmax = np.max([vmax1, vmax2])

    plt.subplot(194)
    plt.imshow(bias1, cmap='gray')
    plt.axis('off')
    plt.subplot(195)
    plt.imshow(bias2, cmap='gray')
    plt.axis('off')
    plt.subplot(196)
    plt.imshow(bias_pred, cmap='gray')
    plt.axis('off')


    img1 = np.divide(img1[0, :, :, 0], bias1)
    img2 = np.divide(img2[0, :, :, 0], bias2)
    img1 = img1 / np.max(img1)
    img2 = img2 / np.max(img2)

    img1 = np.expand_dims(np.expand_dims(img1, axis=0), axis=4)
    img2 = np.expand_dims(np.expand_dims(img2, axis=0), axis=4)

    bias1 = model.predict(img1)
    bias2 = model.predict(img2)
    bias1 = bias1[0, :, :, 0]
    bias2 = bias2[0, :, :, 0]
    bias1 = bias1 / np.mean(bias1)
    bias2 = bias2 / np.mean(bias2)

    plt.figure(1)
    plt.subplot(197)
    plt.imshow(bias1, cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplot(198)
    plt.imshow(bias2, cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplot(199)
    plt.imshow(true_bias, cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.show()
