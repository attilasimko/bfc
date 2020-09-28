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
    rng = 0.15
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
    scale = K.mean(c, axis=(1, 2, 3))
    c = tensorflow.math.subtract(c, scale[:, None, None, None])
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

def build_unet():
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, Dropout, add
    dropout_rate = 0.4
    l2reg = 0.0001
    filt = 2
    kernel_size = 5    

    inp = Input(shape=(64, 64, 1))
    conv1 = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv1 = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)

    conv2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(dropout_rate)(conv2)

    conv3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(dropout_rate)(conv3)

    conv4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    up7 = Conv2D(filt*4, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    #up7 = Concatenate(axis=-1)([conv3,up7])
    up7 = Conv2D(filt*4, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(up7)
    up7 = Conv2D(filt*4, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(up7)
    up7 = BatchNormalization()(up7)
    up7 = Dropout(dropout_rate)(up7)

    up8 = Conv2D(filt*2, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up7))
    # up8 = Concatenate(axis=-1)([conv2,up8])
    up8 = Conv2D(filt*2, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(up8)
    up8 = Conv2D(filt*2, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(up8)
    up8 = BatchNormalization()(up8)
    up8 = Dropout(dropout_rate)(up8)

    up9 = Conv2D(filt, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up8))
    #up9 = Concatenate(axis=-1)([conv1,up9])
    up9 = Conv2D(filt, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(up9)
    up9 = Conv2D(filt, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(up9)
    up9 = BatchNormalization()(up9)
    up9 = Dropout(dropout_rate)(up9)

    x = Conv2D(1, 1, padding = 'same')(up9)
    # x = Concatenate(axis=-1)([inp,x])
    # x = Conv2D(1, 1, padding = 'same')(x)

    # x = Lambda(subtract_layer)([inp, x])
    
    # x = Lambda(normalize_mean)(x)
    # x = Lambda(output_range)(x)


    return Model(inputs=[inp], outputs=[x])

def build_model():
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, UpSampling2D, Concatenate, Dropout, add
    dropout_rate = 0.2
    l2reg = 0.0001
    filt = 4
    kernel_size = 5
    
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

    return Model(inputs=[im1, im2], outputs=[out1, reg1, reg2])

if GPU >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

noise_size = 0.02
orig=cv2.resize(mpimg.imread('orig.png')[:, :, 0], dsize=(64, 64), interpolation=cv2.INTER_LANCZOS4)
orig = np.array(np.interp(orig, (orig.min(), orig.max()), (0, 1)), dtype=np.float32)
noisy = orig + np.random.normal(0, noise_size, size=np.shape(orig))
noisy = np.array(np.interp(noisy, (noisy.min(), noisy.max()), (0, 1)), dtype=np.float32)
# np.savez('/home/attila/out' + str(GPU) + '/_data', orig = orig, noisy = noisy)
print(str(psnr(noisy, orig)))

n_epochs = 50
n_its = 1000
batch_size = 1
lr = 0.0001
dataset = np.repeat(np.expand_dims(np.expand_dims(orig, axis=2), axis=0), batch_size, axis=0)
noisy_dataset = np.repeat(np.expand_dims(np.expand_dims(noisy, axis=2), axis=0), batch_size, axis=0)
zeros = np.zeros_like(dataset)


# Correct images.
model = build_unet()
model.compile(loss=['mean_squared_error'], optimizer=tensorflow.keras.optimizers.Adam(lr))

# implicit = build_implicit()
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
        # x, _ = IN[idx]
        # dataset = x[0] + np.random.normal(0, noise_size, size=np.shape(x[0]))
        # dataset = np.array(np.interp(dataset, (dataset.min(), dataset.max()), (0, 1)), dtype=np.float32)
        noise = np.random.normal(0, noise_size, size=np.shape(dataset))
        full_data = dataset + noise
        full_data = np.array(np.interp(full_data, (full_data.min(), full_data.max()), (0, 1)), dtype=np.float32)
        hist = model.fit(full_data, noise, verbose=0)
        # hist = implicit.fit([full_data, dataset], [noise, zeros, zeros], verbose=0)
        loss.append(hist.history['loss'][0])

    print('Epoch ' + str(epoch) + ':    ' + str(np.round(np.mean(loss), 10)))

    filename = "C:/Users/attil/Documents/DRS/out/orig_%d.png" % (epoch + 1)
    #filename = '/home/attila/out' + str(GPU) + '/orig_%d.png' % (epoch + 1)
    #model.save('/home/attila/out' + str(GPU) + '/%d.h5' % (epoch + 1))
    plt.figure(figsize=(15, 5))
    #plt.subplot(2, 3, 1)
    # plt.imshow(noise[0, :, :, 0])
    # plt.colorbar()
    # plt.title('MV: ' + str(np.mean((noise[0, :, :, 0])**2)))
    # plt.axis('off')
    # plt.subplot(2, 3, 2)
    # img = implicit.predict([full_data, dataset])
    # plt.title('MSE: ' + str(np.mean((img[0][0, :, :, 0] - noise[0, :, :, 0])**2)))
    # plt.imshow(img[0][0, :, :, 0] - noise[0, :, :, 0], cmap='bwr', vmin=-0.1, vmax=0.1)
    # plt.colorbar()
    # plt.axis('off')
    # plt.subplot(2, 3, 3)
    # plt.hist(np.ndarray.flatten(noise[0, :, :, 0]))
    plt.subplot(1, 4, 1)
    plt.imshow(orig)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(noisy)
    plt.colorbar()
    plt.title('Orig PSNR: ' + str(psnr(noisy, orig)))
    plt.axis('off')
    plt.subplot(1, 4, 3)
    img = model.predict(noisy_dataset)
    img = noisy - img[0, :, :, 0]
    plt.imshow(img)
    plt.colorbar()
    plt.title('New PSNR: ' + str(psnr(img, orig)))
    plt.axis('off')
    print(str(psnr(img, orig)))
    plt.subplot(1, 4, 4)
    img = model.predict(dataset)
    plt.hist(np.ndarray.flatten(img[0, :, :, 0]))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')
    tensorflow.keras.backend.clear_session()