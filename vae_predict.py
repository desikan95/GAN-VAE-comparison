from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import numpy
import matplotlib.pyplot

encoder = tensorflow.keras.models.load_model("VAE_encoder.h5", compile=False)
decoder = tensorflow.keras.models.load_model("VAE_decoder.h5", compile=False)

# Preparing MNIST Dataset
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0

x_test = numpy.reshape(x_test, newshape=(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1))

encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)

# def save_imgs(epoch):
r, c = 5, 5
# noise = np.random.normal(0, 1, (r * c, 100))
# gen_imgs = generator.predict(noise)

# Rescale images 0 - 1
decoded_data = 0.5 * decoded_data + 0.5
import matplotlib.pyplot as plt
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
  for j in range(c):
      axs[i,j].imshow(decoded_data[cnt, :,:,0], cmap='gray')
      axs[i,j].axis('off')
      cnt += 1
fig.savefig("VAE_reconstructed_mnist.png")
plt.close()