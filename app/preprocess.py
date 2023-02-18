import math
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import shutil
import os
import gdown

from fastapi import HTTPException

from init import GAUSSIAN_SMOOTH_FILTER_SIZE, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT, GDRIVE_ID


def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    return imgGrayscale, imgThresh


def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    return imgValue


def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement,
                                 iterations=10)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement,
                                   iterations=10)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat


def rotation_angle(linesP):
    angles = []
    for i in range(0, len(linesP)):
        l = linesP[i][0].astype(int)
        p1 = (l[0], l[1])
        p2 = (l[2], l[3])
        doi = (l[1] - l[3])
        ke = abs(l[0] - l[2])
        angle = math.atan(doi / ke) * (180.0 / math.pi)
        if abs(angle) > 45:  # If they find vertical lines
            angle = (90 - abs(angle)) * angle / abs(angle)
        angles.append(angle)

    angles = list(filter(lambda x: (abs(x > 3) and abs(x < 15)), angles))
    if not angles:  # If the angles is empty
        angles = list([0])
    angle = np.array(angles).mean()
    return angle


def rotate_LP(img, angle):
    height, width = img.shape[:2]
    ptPlateCenter = width / 2, height / 2
    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotationMatrix, (width, height))
    return rotated_img


def Hough_transform(threshold_image, nol=6):
    h, w = threshold_image.shape[:2]
    linesP = cv2.HoughLinesP(threshold_image, 1, np.pi / 180, 50, None, 50, 10)
    dist = []
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        d = math.sqrt((l[0] - l[2]) ** 2 + (l[1] - l[3]) ** 2)
        if d < 0.5 * max(h, w):
            d = 0
        dist.append(d)
    dist = np.array(dist).reshape(-1, 1, 1)
    linesP = np.concatenate([linesP, dist], axis=2)
    linesP = sorted(linesP, key=lambda x: x[0][-1], reverse=True)[:nol]

    return linesP


def crop_n_rotate_LP(cropped_LP):
    cropped_LP_copy = cropped_LP.copy()
    imgGrayscaleplate, imgThreshplate = preprocess(cropped_LP)
    canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=2)
    linesP = Hough_transform(dilated_image, nol=6)
    for i in range(0, len(linesP)):
        l = linesP[i][0].astype(int)
    angle = rotation_angle(linesP)
    rotate_thresh = rotate_LP(imgThreshplate, angle)
    LP_rotated = rotate_LP(cropped_LP, angle)
    return angle, rotate_thresh, LP_rotated


def filter_contours(cont):
    s = cv2.contourArea(cont[0])
    u = s / 24
    l = s / 100
    filtered_cont = []
    for c in cont:
        if l < cv2.contourArea(c) < u and len(filtered_cont) != 8:
            filtered_cont.append(c)
    if len(filtered_cont) < 8:
        raise HTTPException(status_code=404, detail="Was not able to find all numbers in LP, maybe angle of the image is bad!")
    return list(filtered_cont)


def get_patches(img):
    cr_LP = crop_n_rotate_LP(img)
    cont, hier = cv2.findContours(cr_LP[-2], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:12]
    cont = filter_contours(cont)
    patches = []
    xs = []
    for c in cont:
        x,y,w,h = cv2.boundingRect(c)
        y,h = 0, cr_LP[-2].shape[0]
        div = int(h/10)
        y += div
        h -= div
        patches.append(Image.fromarray(cr_LP[2][y:y+h, x:x+w]))
        xs.append(x)
    patches = [patch for _, patch in sorted(zip(xs, patches))]
    return patches


def store_uploaded_images(
    images: list, dest_path, max_retries: int = 10
) -> bool:
    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)
    stored = []
    for image in images:
        img_fp = str(dest_path / image.filename)

        # due to linux VM disk, sometimes an OSError raises when trying to write
        # this is a workaround using 'max_retries' and catching 'OSError' exception
        for _ in range(max_retries):
            try:
                with open(img_fp, "wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)
                stored.append(True if os.path.exists(img_fp) else False)
                break
            except OSError:
                pass
    return True if all(stored) else False


def download_from_google_drive(dest):
    for name, id in GDRIVE_ID.items():
        output = Path(dest) / name
        if output.name not in os.listdir(output.parent):
            print(name)
            gdown.download(id=id, output=str(output), quiet=False)
    return dest
