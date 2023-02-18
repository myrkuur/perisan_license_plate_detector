# Perisan License Plate Detector
Using yolov7 to first detect LP and the using swin transformer to detect characters.

## Introduction 
### Training
Training was done on google colab gpus, training notebooks can be found at `src/app/notebooks`.
Training includes two phases:
1. Detection of LP: For this, a pretrained `YoLoV7` was finetunned for `100` epochs and image size of `640x640` on images of cars and corresponding annotation of LP bounding box.
2. Classifying digits inside LP: For this, a pretrained `Swin Transformer` was finetunned on augmented images of LP digits, it was trained for `4` epochs and kept `20 percent` for testing.
<hr>

### Testing And Evaluation
1. YoLoV7 Evaluation:

    #### Sample of model predictions:
    ![alt text](src\app\runs\test\exp2\test_batch0_pred.jpg)
    #### Model's confusion matrix:
    ![alt text](src\app\runs\test\exp2\confusion_matrix.png)
    #### Model's F1 curve:
    ![alt text](src\app\runs\test\exp2\F1_curve.png)
    #### Model's P Curve:
    ![alt text](src\app\runs\test\exp2\P_curve.png)
    #### Model's PR curve:
    ![alt text](src\app\runs\test\exp2\PR_curve.png)
2. Swin Transformer Evaluation:
    Classifier achieved `99` percent accuracy on test set with `F1_macro` of `~100` percent.
### Web App
Using `FastApi` a web app was designed for more user friendly interactions. After running the app you can choose the file with image of a car and see the results, **Note:** that if you are running the app for the first time it will take a bit because it needs to download trained models from google drive.
<hr><hr>

## How to install
1. Clone the repo into your local device.
2. Move to clone project directory (you should be in `perisan_license_plate_detector`)
3. Create conda virtual environment using command below: (you can change environment name to what you like, here we choose `LP_detector`)
    ```
    conda create -n LP_detector python=3.7 anaconda
    ```
4. Activate your environment using command below:
    ```
    conda activate LP_detector
    ```
5. Install requirements using command below:
    ```
    pip install -r requirements.txt
    ```
6. Move to `/src/app` using command below:
    ```
    cd src/app
    ```
7. Run the following command to start the app:
    ```
    uvicorn main:app --reload
    ```
8. After app starts, open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.
<hr><hr>

## Challenges And Improvement Ideas:
1. I believe there are more promising arch to try, for example `RCNN` is a good replacement for `YoLov7` and may worth a shot, also playing with hyper parameters and augmenting might also help.
2. Main issue of the current pipeline is detection of digits inside LP which currently is achieved using `cv2.findContours`. There are not sufficient implementation of persian ocr and a better approach might be training a separate object detection model to extract digit from LP although that will cause more inference time.
