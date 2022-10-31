# -*- coding: utf-8 -*-
import os
import cv2
import h5py
import numpy
from pathlib import Path


# BORDER_CUT = 8
BLOCK_STEP = 16
BLOCK_SIZE = 32
class DataPreprocess: 

   
    
    train_data = []
    train_label = []

    def test_resize_crop_coordinates(shapeX,shapeY):
            # produce Random_Crop random coordinate to crop training img
            points_x = numpy.random.randint(0, min(shapeX, shapeY) - 32, 30)
            points_y = numpy.random.randint(0, min(shapeX, shapeY) - 32, 30)
            return (points_x,points_y)   

    def train_resize_crop_coordinates(shape_x,shape_y):
            width_num = int((shape_x - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP)
            height_num = int((shape_y - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP)
            return (width_num,height_num)       

    def train_random_crop(hr_patch,lr_patch):
            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            lr = numpy.zeros((1, 32, 32), dtype=numpy.double)
            hr = numpy.zeros((1, 20, 20), dtype=numpy.double)

            lr[0, :, :] = lr_patch
            hr[0, :, :] = hr_patch[6: -6, 6: -6]
            return (lr,hr)

    def prepare_test_data(self,_path):
        names = os.listdir(_path)
        names = sorted(names)
        nums = names.__len__()

        test_data = []
        test_label = []

        test_data = numpy.zeros((nums * 30, 1, 32, 32), dtype=numpy.double)
        test_label = numpy.zeros((nums * 30, 1, 20, 20), dtype=numpy.double)

        for i in range(nums):
            name = _path + names[i]
            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            shape = hr_img.shape

            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]

            # two resize operation to produce training data and labels
            lr_img = cv2.resize(hr_img, (int(shape[1] / 2), int(shape[0] / 2)))
            lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

            (points_x,points_y) = self.test_resize_crop_coordinates(shape[0],shape[1])

            for j in range(30):
                lr_patch = lr_img[points_x[j]: points_x[j] + 32, points_y[j]: points_y[j] + 32]
                hr_patch = hr_img[points_x[j]: points_x[j] + 32, points_y[j]: points_y[j] + 32]
                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                test_data[i * 30 + j, 0, :, :] = lr_patch
                test_label[i * 30 + j, 0, :, :] = hr_patch[6: -6, 6: -6]
                

        return test_data, test_label



    def prepare_train_data(self,_path):
        names = os.listdir(_path)
        names = sorted(names)
        nums = names.__len__()

        for i in range(nums):
            name = _path + names[i]
            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]
            shape = hr_img.shape

            # two resize operation to produce training data and labels
            lr_img = cv2.resize(hr_img, (int(shape[1] / 2), int(shape[0] / 2)))
            lr_img = cv2.resize(lr_img, (shape[0], shape[1]))
            
            (width_num, height_num) = self.train_resize_crop_coordinates(hr_img,shape[0],shape[1])

            for k in range(width_num):
                for j in range(height_num):
                    x = k * BLOCK_STEP
                    y = j * BLOCK_STEP
                    hr_patch = hr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
                    lr_patch = lr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
                    (lr,hr) = self.train_random_crop(hr_patch,lr_patch)
                    self.train_data.append(lr)
                    self.train_label.append(hr)

        self.train_data = numpy.array(self.train_data, dtype=float)
        self.train_label = numpy.array(self.train_label, dtype=float)
        return (self.train_data, self.train_label)


    def write_hdf5(data, labels, output_filename):
        x = data.astype(numpy.float32)
        y = labels.astype(numpy.float32)
        with h5py.File(output_filename, 'w') as h:
            h.create_dataset('data', data=x, shape=x.shape)
            h.create_dataset('label', data=y, shape=y.shape)


    if __name__ == '__main__':
        orginalPath = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())
        trainPath = f'{orginalPath}/data/raw/final/srcnn/train/'
        testPath = f'{orginalPath}/data/raw/final/srcnn/test/'
        (data,label) = prepare_train_data(trainPath)
        write_hdf5(data,label,f'{orginalPath}/data/processed/train.h5')
        (data2,label2) = prepare_test_data(testPath)
        write_hdf5(data2,label2,f'{orginalPath}/data/processed/test.h5')
     

# dvc run -n data_preparation_for_training -d data/raw/final/srcnn/train/ -d data/raw/final/srcnn/test/ -o data/processed/train.h5 -o data/processed/test.h5 python src/data/data_preparation.py