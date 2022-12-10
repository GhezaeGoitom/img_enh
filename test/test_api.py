from pathlib import Path
from fastapi.testclient import TestClient
import sys
import os
import yaml
import io
import imghdr
from datetime import datetime
from PIL import Image


original_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.append(original_path)

print(original_path)

from src.service.main import app

client = TestClient(app)

# if __name__ == "__main__":
# testing the main route
def test_main_app():
    response = client.get("/")
    assert response.status_code == 200

# testing retrieve information
def test_info():
    response = client.get("/apps/srcnn/info")
    assert response.status_code == 200

# test models that has been builded
def test_stored_weights():
    response = client.get("/apps/srcnn/models")
    assert response.status_code == 200
    assert response.json()["data"] == retrieve_model_names()

# testing moddle size of kernels, because the first and last are the same
def test_middle_kernel_sizes():
    response = client.get("/apps/srcnn/middle-kernels")
    assert response.status_code == 200
    assert response.json()["data"] == retrieve_middle_sizes_of_kernel()

# testing downloading  weight
def test_weight():
    response = client.get("/apps/srcnn/weight/srcnn.h5")
    assert response.status_code == 200

# testing enhancing image
def test_enhanced_image():
    sampleImagePath = f"{original_path}/data/raw/a.jpg"
    with open(sampleImagePath,'rb') as imgs:
        response = client.post("/apps/srcnn/enhance?model_name=srcnn.h5&middle_kernel_size=1",
        files={"file": ("a.jpg", imgs, "image/jpeg")}
        ) 
    assert response.status_code == 200


 # testing enhancing image
# def test_enhanced_image_in_png():
#     sampleImagePath = f"{original_path}/data/raw/t.png"
#     with open(sampleImagePath,'rb') as imgs:
#         response = client.post("/apps/srcnn/enhance?model_name=srcnn.h5&middle_kernel_size=1",
#         files={"file": ("a.png", imgs, "image/png")}
#         ) 
#     assert response.json()["status-code"] == 400          




def retrieve_model_names():
    models = []
    for(_,_,files) in os.walk(f"{original_path}/src/weights/"):
        for filename in files:
            if ".h5" in filename:
                models.append(filename)
    return models


def retrieve_middle_sizes_of_kernel():
    middle = []
    params = yaml.safe_load(open(f'{original_path}/src/models/params.yaml'))["learn"]["kernel_size"]
    for param in params:
        middle.append(param[1])
    return middle

def create_image(file):
    filename = ''.join(datetime.now().isoformat())
    image_bytes = Image.open(io.BytesIO(file)).convert("RGB")
    imagePath = f"{str(Path(original_path).parent.absolute())}/data/processed/response/{filename}.jpg"
    image_bytes.save(imagePath)
    image_type = imghdr.what(imagePath)
    return image_type      