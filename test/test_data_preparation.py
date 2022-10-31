import os
import sys
import cv2
from matplotlib import test
import pytest

dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dir, "..", "src", "data"))
from DataPreparation import DataPreprocess

MAX_SHAPE = 488

@pytest.fixture
def getImage():
    return cv2.imread("test_img.JPG",cv2.IMREAD_GRAYSCALE)

@pytest.mark.parametrize("shape_x,shape_y",[
    (300,300),
    (300,400),
    (400,300),
    (448,448)]) 
    
def test_test_resize_crop_coordinates(shape_x,shape_y):
    (points_x,points_y) = DataPreprocess.test_resize_crop_coordinates(shape_x,shape_y)
    assert points_x.size < MAX_SHAPE
    assert points_y.size < MAX_SHAPE


@pytest.mark.parametrize("train_shape_x,train_shape_y",[
    (300,300),
    (448,448)])  

def test_train_random_crop(getImage,train_shape_x,train_shape_y):
    (width_num,height_num) = DataPreprocess.train_resize_crop_coordinates(train_shape_x,train_shape_y)
    assert width_num > 0
    assert height_num > 0
 
def test_train_resize_crop_coordinates(getImage):
    (lr,hr) = DataPreprocess.train_random_crop(getImage,getImage)
    assert lr.size > 0
    assert hr.size > 0
