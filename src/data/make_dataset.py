from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pyunpack import Archive
import numpy
import h5py
import os
from random import choice
import shutil
import data_preparation as data_preparation


def main(originalPath):
    Archive(f'{originalPath}/data/raw/dataset.zip').extractall(f'{originalPath}/data/raw/final/')
    (trainPath,testPath) = split_dataset(originalPath)
    (data,label) = data_preparation.prepare_train_data(trainPath)
    write_hdf5(data,label,f'{originalPath}/data/processed/train.h5')
    (data2,label2) = data_preparation.prepare_test_data(testPath)
    write_hdf5(data2,label2,f'{originalPath}/data/processed/test.h5')



#make dir
def make_folder(path="output"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def split_dataset(path):
    i = 0
    imgs = []
    
    #setup ratio
    trainRatio = 0.8
    testRatio = 0.2
    
    #setup dir names
    imagePath = f'{path}/srcnn/' # dir where the splitted images is going to be stored
    crsPath = f'{path}/images/images/' #dir where the original images stored
    
    # path for specific paths
    trainPath = f'{imagePath}/train'
    testPath = f'{imagePath}/test'

    trainimagePath = make_folder(trainimagePath)
    testimagePath = make_folder(testimagePath)
    
    for (_, _, files) in os.walk(crsPath):
        for filename in files:
        # imgs.append(filename)
                if i > 100:
                    break
                else:
                    i = i+1  
                    imgs.append(filename)


    #counting range for cycles
    countForTrain = int(len(imgs)*trainRatio)
    countFortest = int(len(imgs)*testRatio)
    print("training images are : ",countForTrain)
    print("test images are : ",countFortest)

    #cycle for train dir
    for _ in range(countForTrain):

        fileJpg = choice(imgs) # get name of random image from origin dir
        shutil.copy(os.path.join(crsPath, fileJpg), os.path.join(trainPath, fileJpg))

        #remove files from arrays
        imgs.remove(fileJpg)


    #cycle for test dir   
    for _ in range(countFortest):

        fileJpg = choice(imgs) # get name of random image from origin dir
        shutil.copy(os.path.join(crsPath, fileJpg), os.path.join(testPath, fileJpg))   
        
        #remove files from arrays
        imgs.remove(fileJpg)

    return (trainPath,testPath)    


def write_hdf5(data, labels, output_filename):
    x = data.astype(numpy.float32)
    y = labels.astype(numpy.float32)
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)


if __name__ == '__main__':
    orginalPath = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())
    main(orginalPath)
