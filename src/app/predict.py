import platform
import pathlib
import os
import gdown
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

import cv2

from detect import detect
from preprocess import get_patches, download_from_google_drive
from init import MODEL_DIR, CHAR_CLF_DIR, LP_DETECTOR_DIR

from fastai.vision.all import *


def predict(path):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    download_from_google_drive(MODEL_DIR)
    dets = detect(source=path, weights=LP_DETECTOR_DIR)
    for det in dets:
        x, y, w, h, _, _ = det[0]
        img = cv2.imread(path)
        crop_img = img[int(y):int(h), int(x):int(w)]
        patches = get_patches(crop_img)
        model = load_learner(CHAR_CLF_DIR)
        ls = []
        for patch in patches:
            ls.append(model.predict(patch)[0])
        return{f'Detected LP: {ls[0]} {ls[1]} {ls[2]} {ls[3]} {ls[4]} {ls[5]} | {ls[6]} {ls[7]}'}