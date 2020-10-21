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
sys.path.append("C:/Users/attil/Documents/DRS/drs/")
from MLTK.data import BiasGenerator

model = tensorflow.keras.models.load_model('IN.h5', compile=False)
dataset = 'data.npz'

with np.load(dataset) as npzfile:
    v = npzfile['v']
    g = npzfile['g']
    g_hat = npzfile['g_hat']
    u = v / g
    
    vg_hat = v * g_hat
    vg_hat = vg_hat / np.max(vg_hat)

    plt.figure(1, figsize=(20, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(u, cmap='gray')
    plt.title('u')
    plt.axis('off')
    plt.subplot(2, 5, 2)
    plt.imshow(v, cmap='gray')
    plt.title('v = u * g')
    plt.axis('off')
    plt.subplot(2, 5, 7)
    plt.imshow(g, cmap='gray')
    plt.title('g')
    plt.axis('off')

    plt.subplot(2, 5, 3)
    plt.imshow(vg_hat, cmap='gray')
    plt.title('v * g_hat')
    plt.axis('off')
    plt.subplot(2, 5, 8)
    plt.imshow(g_hat, cmap='gray')
    plt.title('g_hat')
    plt.axis('off')


    vg_hat = np.expand_dims(np.expand_dims(vg_hat, axis=0), axis=4)
    v = np.expand_dims(np.expand_dims(v, axis=0), axis=4)

    F_vg_hat = model.predict(vg_hat)
    F_v = model.predict(v)
    F_vg_hat = F_vg_hat[0, :, :, 0]
    F_v = F_v[0, :, :, 0]
    g_hat_pred = np.divide(F_vg_hat, F_v)

    plt.subplot(2, 5, 4)
    plt.imshow(F_vg_hat, cmap='gray')
    plt.title('F(v * g_hat)')
    plt.axis('off')
    plt.subplot(2, 5, 9)
    plt.imshow(F_v, cmap='gray')
    plt.title('F(v)')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(g_hat_pred, cmap='gray')
    plt.title('F(v * g_hat) / F(v)')
    plt.axis('off')
    plt.show()
