from keras.models import Sequential
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from pathlib import Path
import mlflow
import yaml 
import time
import h5py
import numpy

orginalPath = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())


class TrainModel:

    def read_data(file):
        with h5py.File(file, 'r') as hf:
            data = numpy.array(hf.get('data'))
            label = numpy.array(hf.get('label'))
            train_data = numpy.transpose(data, (0, 2, 3, 1))
            train_label = numpy.transpose(label, (0, 2, 3, 1))
            return train_data, train_label


    # define the SRCNN model
    def srcnnModel(layer1, layer2, layer3):
        
        # define model type
        SRCNN = Sequential()

        mlflow.log_params(
        {
            "layer one kernel size": layer1,
            "layer two kernel size": layer2,
            "layer three kernel size": layer3,
        })
        
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


    def train(self,layer1,layer2,layer3):
        # Measure training time
        start_time = time.time()

        srcnn_model = self.srcnnModel(layer1,layer2,layer3)
        print(srcnn_model.summary())
        data, label = self.read_data(orginalPath+"/data/processed/train.h5")
        val_data, val_label = self.read_data(orginalPath+"/data/processed/test.h5")

        checkpoint = ModelCheckpoint(f"{orginalPath}/src/weights/SRCNN_weight_{layer2}.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='min')
        callbacks_list = [checkpoint]

        history = srcnn_model.fit(data, label, batch_size=64, validation_data=(val_data, val_label),
                        callbacks=callbacks_list, shuffle=True, epochs=1, verbose=0)

    # end time
        end_time = time.time()

    # loging to ml flow
        mlflow.log_metrics(
        {
            "accuracy": history.history['loss'][0],
            "val_accuracy": history.history['val_loss'][0],
            "train time": end_time-start_time
        }

    )    

    # saving the matric in text
        with open(f'{orginalPath}/src/metrics/train_metric.txt', 'a') as fl:
            fl.write(f"model_with_layer: 9,{layer2},5 \n accuracy: {history.history['loss'][0]}, \n val_accuracy: {history.history['val_loss'][0]},  \n training_time: {end_time - start_time} \n")
        print("done.")

        return (history.history['loss'][0],history.history['val_loss'][0],end_time-start_time)
        

    def train_experiment_parameters():
        rows, cols = (3, 3)
        arr = [[0 for _ in range(cols)] for _ in range(rows)]

        return arr


    if __name__ == "__main__":
        params = yaml.safe_load(open(f'{orginalPath}/src/models/params.yaml'))["learn"]["kernel_size"]
        for row in params:
            # Start mlflow run
            mlflow.start_run()
            train(row[0],row[1],row[2])
            # End mlflow run
            mlflow.end_run()


    # dvc run -n training -d src/models/params.yaml -d data/processed/train.h5 -d data/processed/test.h5 -M src/metrics/train_metric.txt -o src/weights/SRCNN_weight_1.h5 -o src/weights/SRCNN_weight_3.h5 -o src/weights/SRCNN_weight_5.h5 python src/models/train_model.py

