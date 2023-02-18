from fastapi import FastAPI, UploadFile

from predict import predict
from preprocess import store_uploaded_images
from init import UPLOAD_DIR


app = FastAPI()


@app.post("/LP_detector")
def detect_license_plate(image: UploadFile):
    store_uploaded_images([image], UPLOAD_DIR)
    return predict(str(UPLOAD_DIR / image.filename))
