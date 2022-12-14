from fastapi import FastAPI, File, Response,Request,status, Depends,HTTPException
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse,JSONResponse
from pydantic import BaseModel,ValidationError,validator
from http import HTTPStatus
from functools import wraps
from datetime import datetime
import yaml
import random
import cv2
import io
import string
from pathlib import Path
import sys
import os
import imghdr
import json



original_path = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())
sys.path.append(original_path)
from src.models import PerformSR as sr

middle_kernels = [1,3,5]
model_names = ['srcnn.h5','srcnn2.h5','srcnn3.h5']


app = FastAPI(
    title="Image Enhancement",
    description="Image Enhancement using SRCNN",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Construct a JSON response for an endpoint's results
def construct_response(wrapped_func):
    @wraps(wrapped_func)
    def wrap(request: Request, *args, **kwargs):
        results = wrapped_func(request, *args, **kwargs)

        # return if the response is file
        if results["response_type"] == "file":
            return  FileResponse(path= results["path"],media_type=results["c_t"],filename= results["name"])
        
        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,  # pylint: disable=protected-access
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        

        return response

    return wrap


class EnhanceModel(BaseModel):
    model_name: str
    middle_kernel_size: int

    @validator('model_name')
    def model_is_in_h5_format(cls, v):
        if '.h5' not in v:
            raise ValueError('please choose .h5 extension type model')
        return v

    @validator('middle_kernel_size')
    def kernel_size_must_be(cls,v):
        if v not in [1,3,5]:
            raise ValueError('please choose correct middle kernel size')
        return v


class WeightFileModel(BaseModel):
    model_name: str    


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )    

@app.get('/',tags=["General"])
@construct_response
def apps( request: Request,response: Response):
    return {
        "response_type": "text",
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"result": "app started"}
    }

@app.get('/apps/srcnn/info',tags=["General"])
@construct_response
def info( request: Request,
    response: Response,):
    response = {
        "response_type": "text",
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"project-name":"Image Enhancement using SRCNN",
        "authors": ["Ghezae Goitom","Yonas Babulet"]},
    }
    return response

@app.get('/apps/srcnn/models',tags=["Models"])
@construct_response
def models( request: Request,
    response: Response):
    models = []
    for(_,_,files) in os.walk(f"{original_path}/src/weights/"):
        for filename in files:
            if ".h5" in filename:
                models.append(filename)
    return {
        "response_type": "text",
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": models
    }


@app.get('/apps/srcnn/middle-kernels',tags=["Parameter"])
@construct_response
def kernels( request: Request,
    response: Response):
    middle = []
    params = yaml.safe_load(open(f'{original_path}/src/models/params.yaml'))["learn"]["kernel_size"]
    for param in params:
        middle.append(param[1])    
    return {
        "response_type": "text",
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": middle
    } 
       

@app.post('/apps/srcnn/enhanceByParams', tags=["Prediction"])
@construct_response
def enhance( request: Request,
    response: Response,enhance: EnhanceModel = Depends(), file: bytes = File(...)):
    
    if enhance.model_name not in model_names:
        return {
        "response_type": "text",   
        "message": "Model not found",
        "status-code": HTTPStatus.NOT_FOUND}

    if enhance.middle_kernel_size not in middle_kernels:
        return {
        "response_type": "text",    
        "message": "Kernel size not found",
        "status-code": HTTPStatus.NOT_FOUND}

    filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    image_bytes = Image.open(io.BytesIO(file)).convert("RGB")
    imagePath = f"{original_path}/data/raw/request/{filename}.jpg"
    image_bytes.save(imagePath)
    image_type = imghdr.what(imagePath)
    (x,y) = image_bytes.size
    
    if x == 0 and y == 0 :
        return {
        "response_type": "text",    
        "message": "please choose valid image",
        "status-code": HTTPStatus.BAD_REQUEST}

    if image_type not in ["jpeg","jpg"]:
        return {
        "response_type": "text",    
        "message": "please choose valid image type of jpeg",
        "status-code": HTTPStatus.BAD_REQUEST}
    
    try:
       
        path = sr.performSR(cv2.imread(imagePath),original_path,f"{filename}.jpg")
        response =  {
        "response_type": "file",
        "c_t": "image/jpeg",
        "path": path,
        "name": f"{filename}.jpg"}
    except Exception as ex:
        response =  {
            "response_type": "text",
            "message": ex,
            "status-code": HTTPStatus.BAD_REQUEST
        }            
    return response

@app.post('/apps/srcnn/enhance', tags=["Prediction"])
def enhance(file: bytes = File(...)):

    filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    image_bytes = Image.open(io.BytesIO(file)).convert("RGB")
    imagePath = f"{original_path}/data/raw/request/{filename}.jpg"
    image_bytes.save(imagePath)
    image_type = imghdr.what(imagePath)
    (x,y) = image_bytes.size
    path = sr.performSR(cv2.imread(imagePath),original_path,f"{filename}.jpg")
    print(path)
    return FileResponse(path= path,media_type="image/jpeg",filename= f"{filename}.jpg")


@app.get('/apps/srcnn/weight/{weight_name}',tags=["Models"])
@construct_response
def Weight( request: Request,
    response: Response,weight_name: str):
    weight_path = f"{original_path}/src/weights/{weight_name}"
    if os.path.exists(weight_path) == False:
        response = {
        "response_type": "text",    
        "message": "Weight not found",
        "status-code": HTTPStatus.NOT_FOUND,
        "method": request.method
        }
        return response
    return {
        "response_type": "file",
        "c_t": "application/octet-stream",
        "path": weight_path,
        "name": weight_name}
    