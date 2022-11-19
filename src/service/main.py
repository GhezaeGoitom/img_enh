from fastapi import FastAPI, File, Response,Request,status, Depends,HTTPException
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse,JSONResponse
from pydantic import BaseModel
from http import HTTPStatus
import yaml
import random
import cv2
import io
import string
from pathlib import Path
import sys
import os
import json



original_path = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())
sys.path.append(original_path)

middle_kernels = [1,3,5]
model_names = ['srcnn.h5','srcnn2.h5','srcnn3.h5']

from src.models import PerformSR as sr
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


class EnhanceModel(BaseModel):
    model_name: str
    middle_kernel_size: int

class WeightFileModel(BaseModel):
    model_name: str


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )    

@app.get('/',tags=["General"])
def apps():
    return {"result": "app started"}

@app.get('/apps/srcnn/health')
def health():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "Project Name": "Image Enhancement using SRCNN",
        "data": {"authors": ["Ghezae Goitom","Yonas Babulet"]},
    }
    
    return response

@app.get('/apps.srcnn.models')
async def models():
    models = []
    for(_,_,files) in os.walk(f"{original_path}/src/weights/"):
        for filename in files:
            if ".h5" in filename:
                models.append(filename)
    result = json.dumps({"models":models})
    return Response(result)


@app.get('/apps.srcnn.middle-kernels')
async def kernels():
    middle = []
    params = yaml.safe_load(open(f'{original_path}/src/models/params.yaml'))["learn"]["kernel_size"]
    for param in params:
        middle.append(param[1])
    result = json.dumps({"middle_kernels":middle})    
    return Response(result)    

@app.post('/apps.srcnn.enhance')
async def enhance(enhance: EnhanceModel = Depends(), file: bytes = File(...)):
    if enhance.model_name not in model_names:
        raise HTTPException(status_code=404, detail="Model name not found")
    if enhance.middle_kernel_size not in middle_kernels:
        raise HTTPException(status_code=404, detail="Middle kernel size not found")    
    filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    image_bytes = Image.open(io.BytesIO(file)).convert("RGB")
    imagePath = f"{original_path}/data/raw/request/{filename}.jpg"
    image_bytes.save(imagePath)
    path = sr.performSR(cv2.imread(imagePath),original_path,f"{filename}.jpg")
    bytes_io = io.BytesIO()
    img = Image.open(path)
    img.save(bytes_io, format="jpeg")    
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")



@app.post('/apps.srcnn.weight')
async def Weight(weight_ref: WeightFileModel):
    weight_path = f"{original_path}/src/weights/{weight_ref.model_name}"
    if os.path.exists(weight_path) == False:
        raise HTTPException(status_code=404, detail="Weight not found")
    return FileResponse(path= weight_path,media_type='application/octet-stream',filename= weight_ref.model_name)  