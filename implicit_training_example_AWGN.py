import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras.backend as K
import cv2
import argparse

# Code showcasing implicit training, but instead of bias field correction, it is gaussian noise removal.

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--gpu", default='-1', help="Path to output data.")
args = parser.parse_args()
GPU = int(args.gpu)

def output_range(x):
    from tensorflow.keras import backend as K
    rng = 1
    x = tensorflow.where(K.greater(x, -rng), x, -rng*K.ones_like(x))
    x = tensorflow.where(K.less(x, rng), x, rng*K.ones_like(x))
    return x

def psnr(y_true, y_pred):
    mse = np.mean( (y_true - y_pred) ** 2 )
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def implicit_relation(tensor):
    from tensorflow.keras import backend as K
    a = tensor[0]
    b = tensor[1]
    c = tensorflow.math.subtract(a, b)
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
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, UpSampling2D, Concatenate, Dropout, add
    dropout_rate = 0.2
    filt = 2
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

    #x = Lambda(output_range)(x)

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

u=cv2.resize(mpimg.imread('data/orig.png')[:, :, 0], dsize=(64, 64), interpolation=cv2.INTER_LANCZOS4)
u = np.array(np.interp(u, (u.min(), u.max()), (0, 1)), dtype=np.float32)
noise_size = 0.02
b = np.random.normal(0, noise_size, size=np.shape(u))
v = u + b
v = np.array(np.interp(v, (v.min(), v.max()), (0, 1)), dtype=np.float32)

n_epochs = 10
n_its = 500
lr = 0.0001
v_batch = np.expand_dims(np.expand_dims(v, axis=2), axis=0)
zeros_batch = np.zeros_like(v_batch)


# Correct images.
model = build_model()
model.compile(loss=['mean_squared_error'])

implicit = build_implicit()
implicit.compile(optimizer=tensorflow.keras.optimizers.Adam(lr), 
              loss=['mean_squared_error', 'mean_squared_error', 'mean_squared_error'],
              loss_weights=[1, 0.5, 0.5])

for epoch in range(n_epochs):
    loss = []
    for idx in range(int(n_its)):
        b_hat = np.random.normal(0, noise_size, size=np.shape(v_batch))
        vb_hat = v_batch + b_hat
        vb_hat = np.array(np.interp(vb_hat, (vb_hat.min(), vb_hat.max()), (0, 1)), dtype=np.float32)
        hist = implicit.fit([vb_hat, v_batch], [b_hat, zeros_batch, zeros_batch], verbose=0)
        loss.append(hist.history['loss'][0])

    print('Epoch ' + str(epoch+1) + ':    ' + str(np.round(np.mean(loss), 10)))
    
    b_pred = model.predict(v_batch)
    v_pred = v - b_pred[0, :, :, 0]
    v_pred = np.interp(v_pred, (v_pred.min(), v_pred.max()), (0, 1))

    print('Orig PSNR: ' + str(psnr(v, u)))
    print('New PSNR: ' + str(psnr(v_pred, u)))

    filename = "AWGN_epoch_%d.png" % (epoch + 1)
    plt.figure(figsize=(15, 5.5))
    plt.subplot(1, 3, 1)
    plt.imshow(u)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(v)
    plt.title('Orig PSNR: ' + str(psnr(v, u)))
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(v_pred)
    plt.title('New PSNR: ' + str(psnr(v_pred, u)))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')
    tensorflow.keras.backend.clear_session()