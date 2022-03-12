import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


batch_size = 4
depth = 109
height = 104
width = 85
input_channels = 2
input_shape =(batch_size,depth,height,width,input_channels)

x_train = np.random.rand(100,depth,height,width,input_channels)
y_train = np.random.rand(100,1)

x_test = np.random.rand(20,depth,height,width,input_channels)
y_test = np.random.rand(20,1)


def baseline_model():
    model = models.Sequential()
    model.add(layers.Conv3D(filters=2, kernel_size=3, activation='relu', input_shape=input_shape[1:]))
    model.add(layers.MaxPooling3D(pool_size=2))
    model.add(layers.Conv3D(filters=4, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='relu'))
    model.add(layers.Dense(10))

    return model

def train():
    model = baseline_model()
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, 
                        validation_data=(x_test, y_test))


train()