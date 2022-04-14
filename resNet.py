import tensorflow as tf
import numpy as np
import os

def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, (3,3,3), strides=(1,1,1), padding = 'same', data_format = "channels_last")(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv3D(filter, (3,3,3), strides=(1,1,1), padding = 'same', data_format = "channels_last")(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, (3,3,3), strides=(2,2,2), padding = 'same', data_format = "channels_last")(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv3D(filter, (3,3,3), strides=(1,1,1), padding = 'same', data_format = "channels_last")(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv3D(filter, (1,1,1), strides = (2,2,2), data_format = "channels_last")(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNet34(shape):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding3D((3, 3,3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv3D(64, kernel_size=(7,7,7), strides=(2,2,2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(3,3,3), strides=(2,2,2), padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling3D((2,2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
    return model

def train():
    model = ResNet34((109, 104, 85, 2))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, 
                validation_data=(x_vald, y_vald))
    
    model.save('saved_model/resNet')

def read_single_frame(path, depth, height, width):
    with open(path, "rb") as f:
        npData = np.load(f)
        print("reading "+path+" ...")
    frameData = npData.reshape((depth,height,width),order='F') # Using order F to maintain x-y-z order
    return frameData

def read_trajactory(path, field):
    frames = []
    files = os.listdir(path) 
    files.sort()
    for file_ in files:
        f_name = str(file_)
        if field == 1 and 'fld-01' in f_name:
            frames.append(read_single_frame(path+f_name, depth, height, width))
        elif field == 2 and 'fld-02' in f_name:
            frames.append(read_single_frame(path+f_name, depth, height, width))
        # else:
        #     print('Invalid field number')
    frames = np.stack((frames), axis=0)
    return frames

depth = 109
height = 104
width = 85
input_channels = 2

path = 'data/b2yel/'
stericMIF = read_trajactory(path, 1)
elecMIF = read_trajactory(path, 2)
X = np.stack((stericMIF, elecMIF), axis=4)
x_train = X[:700]
x_vald = X[701:901]
x_test = X[901:]
Y = np.ones((1001,1))*0.7809668
y_train = Y[:700]
y_vald = Y[701:901]
y_test = Y[901:]
train()