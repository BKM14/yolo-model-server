from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from ultralytics import YOLO
import io
import tensorflow as tf
from PIL import Image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = YOLO('best.pt')
        print("Model loaded successfully!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

def preprocess_image(image_data: bytes):
    image = Image.open(io.BytesIO(image_data)).convert("L")
    image = image.resize((64, 64))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(file)
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    results = model.predict(source=image, imgsz=640, save=False)
    
    result = results[0]
    
    detections = []
    for box in result.boxes:
        detection = {
            "class_id": int(box.cls[0]),
            "class_name": result.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": {
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3])
            }
        }
        detections.append(detection)
    
    return {
        "detections": detections,
        "image_shape": {
            "width": result.orig_shape[1],
            "height": result.orig_shape[0]
        },
        "inference_time_ms": result.speed['inference']
    }

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}
