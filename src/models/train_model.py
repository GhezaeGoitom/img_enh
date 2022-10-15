from keras.models import Sequential
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from pathlib import Path
import sys
orginalPath = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
print(path)
sys.path.insert(0, path)
from data import train_data_preparation as pd


import pickle
import os
import dotenv


# define the SRCNN model
def srcnnModel(layer1, layer2, layer3):
    
    # define model type
    SRCNN = Sequential()
    
    # add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size = (layer1, layer1), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (layer2, layer2), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (layer3, layer3), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    
    # define optimizer
    adam = Adam(lr=0.0003)
    
    # compile model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def train(layer1,layer2,layer3):
    srcnn_model = srcnnModel(layer1,layer2,layer3)
    print(srcnn_model.summary())
    PROCESSED_DATA_PATH = os.environ.get("PROCESSED_DATA_PATH")
    PROCESSED_TEST_PATH = os.environ.get("PROCESSED_TEST_PATH")
    data, label = pd.read_data(orginalPath+"/data/processed/srcnn_train.h5")
    val_data, val_label = pd.read_data(orginalPath+"/data/processed/srcnn_test.h5")

    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, epochs=1, verbose=0)

    

def train_experiment_parameters():
    rows, cols = (3, 3)
    arr = [[0 for i in range(cols)] for j in range(rows)]
    arr[0] = [9,1,5]
    arr[1] = [9,3,5]
    arr[2] = [9,5,5]
    return arr


if __name__ == "__main__":
    global PROCESSED_DATA_PATH
    global PROCESSED_TEST_PATH
    # dotenv_path = os.path.join(path, '.env')
    # dotenv.load_dotenv(dotenv_path)
    params = train_experiment_parameters()
    for row in params:
        train(row[0],row[1],row[2])