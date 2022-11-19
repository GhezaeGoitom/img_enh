import os
from sre_constants import ASSERT
import sys
import pytest
import pathlib as Path

path = os.path.dirname(__name__)
orginalPath = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())
sys.insert.path(1,os.path.join(path,"..","src","models"))
from image_enhancement.src.models import PredictModel 


@pytest.fixture
def init():
     IMG_NAME = f"{orginalPath}/data/raw/faces.jpeg"
     INPUT_NAME = f"{orginalPath}/data/processed/faces.jpg"
     OUTPUT_NAME = f"{orginalPath}/data/processed/output_faces.jpg"

def test_enhanced_image():
    (bicubic,srcnn) = PredictModel.predict(5)
    #check if the image enhancer is producing false scale
    assert bicubic > 0
    #check if the image enhancer is producing false scale
    assert srcnn > 0
    #check if bicubic scale is constant for same image
    assert bicubic == 36.63356570402663
    #check if srcnn scale is constant for same image
    assert srcnn == 37.83815928321347


# Changes in the input should not affect the output
# def invarainceTest(init):
#     (bicubic1,srcnn1) = PredictModel.predict(5,init.IMG_NAME,init.INPUT_NAME,init.OUTPUT_NAME)
#     (bicubic2,srcnn2) = PredictModel.predict(5,init.IMG_NAME,init.INPUT_NAME,init.OUTPUT_NAME) 
#     assert bicubic1 == bicubic2
#     assert srcnn1 == srcnn2

# # Changes in the input should affect the output
# def directionalTest(init):
#     (bicubic1, srcnn1) = PredictModel.predict(5,init.IMG_NAME,init.INPUT_NAME,init.OUTPUT_NAME)
#     (bicubic2, srcnn2) = PredictModel.predict(5,init.INPUT_NAME,init.INPUT_NAME,init.OUTPUT_NAME)
#     assert bicubic1 != bicubic2
#     assert srcnn1 != srcnn2
