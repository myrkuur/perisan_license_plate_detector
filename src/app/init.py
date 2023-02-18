from pathlib import Path


GDRIVE_ID = {
    "yolov7_best.pt": "18NICGiEg83U9YbJ4heg-Cgem1wfeYjNi",
    "Char_detector_swin.pkl": "1KquDD4rfLM7Re9NLPVf93gS61N3dsdAR"
    }

DIR = Path(__file__).absolute().parent.parent.parent
MODEL_DIR = DIR / 'models'
CHAR_CLF_DIR = MODEL_DIR / 'Char_detector_swin.pkl'
LP_DETECTOR_DIR = MODEL_DIR / 'yolov7_best.pt'
UPLOAD_DIR = DIR / 'uploads'

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5) 
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
