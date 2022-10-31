from keras.models import Sequential
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
import cv2
import numpy
from pathlib import Path

orginalPath = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())

class PredictModel:

    def predict_model(middle_layer):
        # lrelu = LeakyReLU(alpha=0.1)
        SRCNN = Sequential()

        # add model layers
        SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                        activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
        SRCNN.add(Conv2D(filters=64, kernel_size = (middle_layer, middle_layer), kernel_initializer='glorot_uniform',
                        activation='relu', padding='same', use_bias=True))
        SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                        activation='linear', padding='valid', use_bias=True))
        
        # define optimizer
        adam = Adam(lr=0.0003)
        SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
        return SRCNN

    def predict(self,middle_layer,img_name,input_name,output_name):
        srcnn_model = self.predict_model(middle_layer)
        srcnn_model.load_weights(f"{orginalPath}/src/weights/SRCNN_weight_5.h5")
        

    
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        shape = img.shape
        Y_img = cv2.resize(img[:, :, 0], (int(shape[1] / 2), int(shape[0] / 2)), cv2.INTER_CUBIC)
        Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
        img[:, :, 0] = Y_img
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(input_name, img)

        Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = Y_img.astype(float) / 255.
        pre = srcnn_model.predict(Y, batch_size=1) * 255.
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = pre.astype(numpy.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(output_name, img)

        # psnr calculation:
        im1 = cv2.imread(img_name, cv2.IMREAD_COLOR)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
        im2 = cv2.imread(input_name, cv2.IMREAD_COLOR)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
        im3 = cv2.imread(output_name, cv2.IMREAD_COLOR)
        im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

        # saving the matric in file
        with open(f'{orginalPath}/src/metrics/evaluate_metric.txt', 'a') as fl:
            fl.write(f"model_with_layer: 9,5,5 \n bicubic: {cv2.PSNR(im1,im2)}, \n SRCNN : {cv2.PSNR(im1,im3)}\n")
        print("done.")

        return (cv2.PSNR(im1,im2), cv2.PSNR(im1,im3))



    if __name__ == "__main__":
        IMG_NAME = f"{orginalPath}/data/raw/faces.jpeg"
        INPUT_NAME = f"{orginalPath}/data/processed/faces.jpg"
        OUTPUT_NAME = f"{orginalPath}/data/processed/output_faces.jpg"
        predict(5,IMG_NAME,INPUT_NAME,OUTPUT_NAME)


# dvc run -n evaluation -d src/weights/SRCNN_weight_5.h5 -d data/raw/faces.jpeg -M src/metrics/evaluate_metric.txt -o data/processed/faces.jpg -o data/processed/output_faces.jpg python src/models/predict_model.py
