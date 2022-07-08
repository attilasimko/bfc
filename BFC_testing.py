import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('weights/IN.h5', compile=False)
dataset = 'data/data.npz'

with np.load(dataset) as npzfile:
    v = npzfile['v']
    b = npzfile['g']
    b_hat = npzfile['g_hat']
    u = v / b
    
    vb_hat = v * b_hat
    vb_hat = vb_hat / np.max(vb_hat)

    plt.figure(1, figsize=(20, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(u, cmap='gray')
    plt.title('u')
    plt.axis('off')
    plt.subplot(2, 5, 2)
    plt.imshow(v, cmap='gray')
    plt.title('v = u * b')
    plt.axis('off')
    plt.subplot(2, 5, 7)
    plt.imshow(b, cmap='gray')
    plt.title('b')
    plt.axis('off')

    plt.subplot(2, 5, 3)
    plt.imshow(vb_hat, cmap='gray')
    plt.title('v * b_hat')
    plt.axis('off')
    plt.subplot(2, 5, 8)
    plt.imshow(b_hat, cmap='gray')
    plt.title('b_hat')
    plt.axis('off')


    vb_hat = np.expand_dims(np.expand_dims(vb_hat, axis=0), axis=3)
    v = np.expand_dims(np.expand_dims(v, axis=0), axis=3)

    F_vb_hat = model.predict(vb_hat)
    F_v = model.predict(v)
    F_vb_hat = F_vb_hat[0, :, :, 0]
    F_v = F_v[0, :, :, 0]
    b_hat_pred = np.divide(F_vb_hat, F_v)

    plt.subplot(2, 5, 4)
    plt.imshow(F_vb_hat, cmap='gray')
    plt.title('F(v * b_hat)')
    plt.axis('off')
    plt.subplot(2, 5, 9)
    plt.imshow(F_v, cmap='gray')
    plt.title('F(v)')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(b_hat_pred, cmap='gray')
    plt.title('F(v * b_hat) / F(v)')
    plt.axis('off')
    plt.show()
