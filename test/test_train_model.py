import os
import sys
import pytest


path = os.path.dirname(__name__)
sys.path.insert(1,os.path.join(path,"..","src","data"))
from image_enhancement.src.models.TrainModel import TrainModel

def test_train():
    (train_loss,val_loss,time) = TrainModel.train(9,1,5)
    # check if the train loss is accurate
    assert train_loss > 0
    #check if the validation loss is accurate
    assert val_loss > 0
    #check if the model really executed successfuly
    assert time > 0


