import cv2

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.yolov4 import YOLOV4

from what.cli.model import *
from what.utils.file import get_file

what_yolov4_model_list = what_model_list[4:6]

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
WHAT_YOLOV4_MODEL_FILE = what_yolov4_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_YOLOV4_MODEL_URL  = what_yolov4_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_YOLOV4_MODEL_HASH = what_yolov4_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE)):
    get_file(WHAT_YOLOV4_MODEL_FILE,
             WHAT_MODEL_PATH,
             WHAT_YOLOV4_MODEL_URL,
             WHAT_YOLOV4_MODEL_HASH)

# Darknet
model = YOLOV4(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, what_yolov4_model_list[0][WHAT_MODEL_FILE_INDEX]))

# You can also use your own model
# model = YOLOV4(COCO_CLASS_NAMES, "models/yolov4.h5")

while True:
    _, orig_image = cap.read()
    if orig_image is None:
        continue

    # Image preprocessing
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # Run inference
    images, boxes, labels, probs = model.predict(image, 10, 0.4)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes onto the image
    output = draw_bounding_boxes(image, boxes, labels, model.class_names, probs);

    cv2.imshow('YOLOv4 Darknet', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
