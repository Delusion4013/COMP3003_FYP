import tensorflow as tf
import numpy as np
import pandas as pd
import os

depth = 109
height = 104
width = 85
input_channels = 2

instanceCount = 5
split = 0.8

model_name = 'resNet34-' + str(instanceCount)
save_path = '/home/scycw2/training_data/' + model_name + '/'

def read_y_data():
    data = pd.read_csv("data/bf_w4out4pce.csv", sep=",")
    label_list = data['PDB'].values
    y_list = data['pIC50'].values
    return label_list, y_list


def get_4d_couple(parent_path, molecular_name, frame_index):
    if 0 < frame_index < 10:
        mol_field1 = parent_path + tf.get_static_value(molecular_name).decode('utf-8') + "/_fld-01_obj-0" + str(frame_index) + ".npy"
        mol_field2 = parent_path + tf.get_static_value(molecular_name).decode('utf-8') + "/_fld-02_obj-0" + str(frame_index) + ".npy"
    else:
        mol_field1 = parent_path + tf.get_static_value(molecular_name).decode('utf-8') + "/_fld-01_obj-" + str(frame_index) + ".npy"
        mol_field2 = parent_path + tf.get_static_value(molecular_name).decode('utf-8') + "/_fld-02_obj-" + str(frame_index) + ".npy"

    with open(mol_field1, "rb") as f:
        current_mol1 = np.load(f)
    current_mol1 = current_mol1.reshape((109,104,85),order='F')

    with open(mol_field2, "rb") as f:
        current_mol2 = np.load(f)
    current_mol2 = current_mol2.reshape((109,104,85),order='F')
    
    frame = np.stack((current_mol1, current_mol2), axis = 3)
    return frame

def get_5d_list(parent_path, label_list, frame_index):
    frames = []
    for label in label_list:
        for i in range(1,frame_index+1):
            frames.append(get_4d_couple(parent_path, label, i*100+1))
    frames = np.stack((frames), axis=0)
    tf.random.set_seed(12)
    frames = tf.random.shuffle(frames)
    return frames


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, (3,3,3), strides=(1,1,1), padding = 'same', data_format = "channels_last")(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv3D(filter, (3,3,3), strides=(1,1,1), padding = 'same', data_format = "channels_last")(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
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
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv3D(64, kernel_size=(7,7,7), strides=(2,2,2), padding='same')(x_input)
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
    x = tf.keras.layers.Dense(1000, activation = 'sigmoid')(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
    return model

def iter(model, x_train, y_train, x_test, y_test):

    checkpoint_path = save_path+"checkpoints/epoch_{epoch:02d}-cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    csv_output_path = save_path+"log.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(csv_output_path, separator=',', append=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", min_delta=1e-3,patience = 3, verbose = 1)

    # Create a callback that saves the models' weight
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    history = model.fit(x_train, 
                        y_train, 
                        epochs=20, 
                        batch_size=64,
                        validation_data=(x_test, y_test),                        
                        callbacks=[cp_callback,csv_logger])
    
    return model

def train():
    model = ResNet34((109, 104, 85, 2))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mse','mae'])

    # label_list, y_list = read_y_data()
    
    # # Min-max rescaling
    # y_list = (y_list - min(y_list))/(max(y_list)-min(y_list))

    # train_index = int(len(label_list)/10*8)
    
    # tf.random.set_seed(12)
    # label_list_shuffled = tf.random.shuffle(label_list)
    # labels_train = label_list_shuffled[:train_index]
    # labels_test = label_list_shuffled[train_index:]

    # tf.random.set_seed(12)
    # y_list_shuffled = tf.random.shuffle(y_list)
    # y_list_train = y_list_shuffled[:train_index]
    # y_list_test = y_list_shuffled[train_index:]

    # parent_path = "data/b"

    # X_train = get_5d_list(parent_path, labels_train, instanceCount)
    # X_test = get_5d_list(parent_path, labels_test, instanceCount)

    # tf.random.set_seed(12)
    # Y_train = tf.random.shuffle(np.repeat(y_list_train,instanceCount))
    
    # tf.random.set_seed(12)
    # Y_test = tf.random.shuffle(np.repeat(y_list_test,instanceCount))

    model.summary()


    # if not os.path.exists(save_path+"raw_data_10/"):
    #     os.makedirs(save_path+"raw_data_10/")


    # with open(save_path+"raw_data_10/X_train.npy", "wb") as f:
    #     np.save(f, X_train)

    # with open(save_path+"raw_data_10/Y_train.npy", "wb") as f:
    #     np.save(f, Y_train)

    # with open(save_path+"raw_data_10/X_test.npy", "wb") as f:
    #     np.save(f, X_test)

    # with open(save_path+"raw_data_10/Y_test.npy", "wb") as f:
    #     np.save(f, Y_test)

    # model = iter(model, X_train,Y_train,X_test,Y_test)

    # model_path = save_path+'saved_model/'
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # model.save(model_path)

train()