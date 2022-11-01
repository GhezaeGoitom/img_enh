from pathlib import Path
from pyunpack import Archive
import os
from random import choice
import shutil


def main(originalPath):
    make_folder(f'{originalPath}/data/raw/final/')
    Archive(f'{originalPath}/data/raw/dataset.zip').extractall(f'{originalPath}/data/raw/final/')
    (trainPath,testPath) = split_dataset(originalPath)
    


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
    imagePath = f'{path}/data/raw/final/srcnn/' # dir where the splitted images is going to be stored
    crsPath = f'{path}/data/raw/final/images/images/' #dir where the original images stored
    
    # path for specific paths
    trainPath = f'{imagePath}/train'
    testPath = f'{imagePath}/test'

    trainimagePath = make_folder(trainPath)
    testimagePath = make_folder(testPath)
    
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

if __name__ == '__main__':
    orginalPath = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())
    main(orginalPath)
