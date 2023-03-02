'''
Ghamand hi insaan ka sarvanash karta hai - Deepan 
'''

from pickle import NONE
from fastapi import FastAPI,UploadFile,File
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet import MobileNet, preprocess_input
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL = load_model("fruits.h5")
CLASS_NAMES = ["Apple","Banana","Cherry","Dragon Fruit","Mango","Orange","Papaya","Pineapple"]

@app.get("/home")
async def home():
    return "Welcome to the home page"

def read_file_as_image(data) -> image.img_to_array:
    img = image.load_img(BytesIO(data),target_size = (224,224))
    
    img = image.img_to_array(img)
    return img

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)  
):
    img = read_file_as_image(await file.read())
    img = preprocess_input(img)
    prediction = MODEL.predict(img.reshape(1,224,224,3))
    output = np.argmax(prediction)
    predicted_class = CLASS_NAMES[output]
    max_prob = np.max(prediction)
    if max_prob >0.85:
        return {
        'class' : predicted_class,
        'confidence' : float(max_prob)
    }
    else:
        return NONE
    
if __name__== "__main__":
    uvicorn.run(app,host = 'localhost',port = 8000)