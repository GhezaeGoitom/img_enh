# -*- coding: utf-8 -*-
import os
import cv2
import h5py
import numpy
from pathlib import Path




def prepare_test_data(_path):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()

    data = numpy.zeros((nums * 30, 1, 32, 32), dtype=numpy.double)
    label = numpy.zeros((nums * 30, 1, 20, 20), dtype=numpy.double)

    for i in range(nums):
        name = _path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        shape = hr_img.shape

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]

         # two resize operation to produce training data and labels
        lr_img = cv2.resize(hr_img, (int(shape[1] / 2), int(shape[0] / 2)))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        # produce Random_Crop random coordinate to crop training img
        Points_x = numpy.random.randint(0, min(shape[0], shape[1]) - 32, 30)
        Points_y = numpy.random.randint(0, min(shape[0], shape[1]) - 32, 30)

        for j in range(30):
            lr_patch = lr_img[Points_x[j]: Points_x[j] + 32, Points_y[j]: Points_y[j] + 32]
            hr_patch = hr_img[Points_x[j]: Points_x[j] + 32, Points_y[j]: Points_y[j] + 32]

            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            data[i * 30 + j, 0, :, :] = lr_patch
            label[i * 30 + j, 0, :, :] = hr_patch[6: -6, 6: -6]
            # cv2.imshow("lr", lr_patch)
            # cv2.imshow("hr", hr_patch)
            # cv2.waitKey(0)
    return data, label

# BORDER_CUT = 8
BLOCK_STEP = 16
BLOCK_SIZE = 32


def prepare_train_data(_path):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()

    data = []
    label = []

    for i in range(nums):
        name = _path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]
        shape = hr_img.shape
        print('hello')
        print(shape)

 # two resize operation to produce training data and labels
        lr_img = cv2.resize(hr_img, (int(shape[1] / 2), int(shape[0] / 2)))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        width_num = int((shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP)
        height_num = int((shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP)

        for k in range(width_num):
            for j in range(height_num):
                x = k * BLOCK_STEP
                y = j * BLOCK_STEP
                hr_patch = hr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
                lr_patch = lr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]

                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                lr = numpy.zeros((1, 32, 32), dtype=numpy.double)
                hr = numpy.zeros((1, 20, 20), dtype=numpy.double)

                lr[0, :, :] = lr_patch
                hr[0, :, :] = hr_patch[6: -6, 6: -6]

                data.append(lr)
                label.append(hr)

    data = numpy.array(data, dtype=float)
    label = numpy.array(label, dtype=float)
    return (data, label)


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