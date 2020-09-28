import numpy as np
from scipy import interpolate
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import sys
import os
import scipy.io
import scipy.misc
import tensorflow
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras.backend as K
import cv2
import math
from bm3d import bm3d
import argparse

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--gpu", default='-1', help="Path to output data.")
args = parser.parse_args()
GPU = int(args.gpu)

def output_range(x):
    from tensorflow.keras import backend as K
    rng = 0.2
    x = tensorflow.where(K.greater(x, -rng), x, -rng*K.ones_like(x))
    x = tensorflow.where(K.less(x, rng), x, rng*K.ones_like(x))
    return x

def psnr(y_true, y_pred):
    mse = np.mean( (y_true - y_pred) ** 2 )
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def subtract_layer(tensor):
    from tensorflow.keras import backend as K
    a = tensor[0]
    b = tensor[1]
    c = tensorflow.math.subtract(a, b)
    return c

def implicit_relation(tensor):
    from tensorflow.keras import backend as K
    a = tensor[0]
    b = tensor[1]
    c = tensorflow.math.subtract(a, b)
    #scale = K.mean(c, axis=(1, 2, 3))
    #c = tensorflow.math.subtract(c, scale[:, None, None, None])
    return c
    
def implicit_correction(tensor):
    from tensorflow.keras import backend as K
    a = tensor[0]
    b = tensor[1]
    c = tensorflow.math.subtract(a, b)
    c_min = K.min(c, axis=(1, 2, 3))
    c = tensorflow.math.add(c, c_min[:, None, None, None])
    c_max = K.max(c, axis=(1, 2, 3))
    c = tensorflow.math.divide_no_nan(c, c_max[:, None, None, None])
    return c

def build_model():
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, UpSampling2D, Concatenate, Dropout, add
    dropout_rate = 0.2
    l2reg = 0.0001
    filt = 4
    kernel_size = 10
    
    inp = Input(shape=(64, 64, 1))
    x = Conv2D(filt, kernel_size, padding = 'same')(inp)
    x = Conv2D(filt*2, kernel_size, padding = 'same')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filt*4, kernel_size, padding = 'same')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filt*8, kernel_size, padding = 'same')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filt*16, kernel_size, padding = 'same')(x)
    x = Conv2D(1, kernel_size, padding = 'same')(x)

    x = Lambda(output_range)(x)

    return Model(inputs=[inp], outputs=[x])

def build_implicit():
    im1 = Input(shape=(64, 64, 1))
    im2 = Input(shape=(64, 64, 1))
    im1out1 = model(im1)
    im2out1 = model(im2)

    out1 = Lambda(implicit_relation)([im1out1, im2out1])

    # Regularize Image 1
    im1n = Lambda(implicit_correction)([im1, im1out1])
    reg1 = model(im1n)

    # Regularize Image 2
    im2n = Lambda(implicit_correction)([im2, im2out1])
    reg2 = model(im2n)

    return Model(inputs=[im1, im2], outputs=[out1])#, reg1, reg2])

if GPU >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

noise_size = 0.01
orig=cv2.resize(mpimg.imread('orig.png')[:, :, 0], dsize=(64, 64), interpolation=cv2.INTER_LANCZOS4)
orig = np.array(np.interp(orig, (orig.min(), orig.max()), (0, 1)), dtype=np.float32)
orig_noise = np.random.normal(0, noise_size, size=np.shape(orig))
noisy = orig + orig_noise
noisy = np.array(np.interp(noisy, (noisy.min(), noisy.max()), (0, 1)), dtype=np.float32)
# np.savez('/home/attila/out' + str(GPU) + '/_data', orig = orig, noisy = noisy)
print(str(psnr(noisy, orig)))

n_epochs = 100
n_its = 1000
batch_size = 1
lr = 0.000001
noisy_dataset = np.repeat(np.expand_dims(np.expand_dims(noisy, axis=2), axis=0), batch_size, axis=0)
zeros = np.zeros_like(noisy_dataset)


# Correct images.
model = build_model()
model.compile(loss=['mean_squared_error'], optimizer=tensorflow.keras.optimizers.Adam(lr))

implicit = build_implicit()
implicit.compile(optimizer=tensorflow.keras.optimizers.Nadam(lr), 
              loss=['mean_squared_error'],
              loss_weights=[1])
# implicit.compile(optimizer=tensorflow.keras.optimizers.Nadam(lr), 
#               loss=['mean_squared_error', 'mean_squared_error', 'mean_squared_error'],
#               loss_weights=[1, 0.5, 0.5])

              
# IN_ID = '/home/attila/data/DS0044s/IN'
# IN = DataGenerator(IN_ID,
#                     inputs=[['image', False, 'float32']],
#                     outputs=[],
#                     batch_size=batch_size,
#                     shuffle=True)
#
# print(implicit.count_params())
file_id = 0

for epoch in range(n_epochs):
    loss = []
    for idx in range(int(n_its / batch_size)):
        noise = np.random.normal(0, noise_size, size=np.shape(noisy_dataset))
        full_data = noisy_dataset + noise
        full_data = np.array(np.interp(full_data, (full_data.min(), full_data.max()), (0, 1)), dtype=np.float32)
        hist = implicit.fit([full_data, noisy_dataset], [noise], verbose=0)
        loss.append(hist.history['loss'][0])

    print('Epoch ' + str(epoch) + ':    ' + str(np.round(np.mean(loss), 10)))
    
    img = model.predict(noisy_dataset)
    hist_orig = np.ndarray.flatten(orig_noise)
    hist_pred = np.ndarray.flatten(img[0, :, :, 0])
    bins = np.linspace(np.min([hist_orig, hist_pred]), np.max([hist_orig, hist_pred]), 50)
    
    img = noisy - img[0, :, :, 0]
    print(str(psnr(img, orig)))


    filename = "C:/Users/attil/Documents/DRS/out/orig_%d.png" % (epoch + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(orig)
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(noisy)
    plt.title('Orig PSNR: ' + str(psnr(noisy, orig)))
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(img)
    plt.title('New PSNR: ' + str(psnr(img, orig)))
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.hist(hist_orig, bins, alpha=0.5, label='Original')
    plt.hist(hist_pred, bins, alpha=0.5, label='Predicted')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')
    tensorflow.keras.backend.clear_session()