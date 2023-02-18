from pathlib import Path

import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from utils.torch_utils import select_device


def detect(source, weights, imgsz=640):
    set_logging()
    device = select_device('cpu')
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # Second-stage classifier
    # classify = True
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Run inference
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() 
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=True)[0]
        # Apply NMS
        pred = non_max_suppression(pred)
        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)
        #     print(pred)

        # Process detections
        dets = []
        for det in pred:  # detections per image
            p, im0 = path, im0s

            p = Path(p)  # to Path
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                dets.append(det)
        return dets

if __name__ == '__main__':
    with torch.no_grad():
        detect()
