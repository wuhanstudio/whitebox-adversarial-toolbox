import cv2

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes

from what.models.detection.yolox.yolox_x import YOLOX_X
from what.models.detection.yolox.yolox_l import YOLOX_L
from what.models.detection.yolox.yolox_m import YOLOX_M
from what.models.detection.yolox.yolox_s import YOLOX_S

from what.cli.model import *
from what.utils.file import get_file

what_yolox_model_list = what_model_list[9:13]

video = input(f"Please input the OpenCV capture device (e.g. 0, 1, 2): ")

while not video.isdigit():
    video = input(f"Please input the OpenCV capture device (e.g. 0, 1, 2): ")

# Capture from camera
cap = cv2.VideoCapture(int(video))
#cap.set(3, 1920)
#cap.set(4, 1080)

# Check what_model_list for all supported models
index = 0

# Download the model first if not exists
WHAT_YOLOX_MODEL_FILE = what_yolox_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_YOLOX_MODEL_URL  = what_yolox_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_YOLOX_MODEL_HASH = what_yolox_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_YOLOX_MODEL_FILE)):
    get_file(WHAT_YOLOX_MODEL_FILE,
             WHAT_MODEL_PATH,
             WHAT_YOLOX_MODEL_URL,
             WHAT_YOLOX_MODEL_HASH)

if index == 0:
    model = YOLOX_X(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOX_MODEL_FILE))

if index == 1:
    model = YOLOX_L(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOX_MODEL_FILE))

if index == 2:
    model = YOLOX_M(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOX_MODEL_FILE))

if index == 3:
    model = YOLOX_S(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOX_MODEL_FILE))

while True:
    _, orig_image = cap.read()
    if orig_image is None:
        continue

    # Image preprocessing
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # Run inference
    images, boxes, labels, probs = model.predict(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes onto the image
    output = draw_bounding_boxes(image, boxes, labels, model.class_names, probs);

    cv2.imshow('YOLOX', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
