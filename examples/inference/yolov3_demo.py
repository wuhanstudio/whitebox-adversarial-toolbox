import cv2

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.yolov3 import YOLOV3

from what.cli.model import *
from what.utils.file import get_file

what_yolov3_model_list = what_model_list[0:4]

# Capture from camera
cap = cv2.VideoCapture(0)
#cap.set(3, 1920)
#cap.set(4, 1080)

# Check what_model_list for all supported models
index = 1

# Download the model first if not exists
if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX])):
    get_file(what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX],
                WHAT_MODEL_PATH,
                what_yolov3_model_list[index][WHAT_MODEL_URL_INDEX],
                what_yolov3_model_list[index][WHAT_MODEL_HASH_INDEX])

# MobileNet
model = YOLOV3(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX]))

# You can also use your own model
# model = YOLOV3(COCO_CLASS_NAMES, "models/yolov3.h5")
# model = YOLOV3(COCO_CLASS_NAMES, "models/yolov3_mobilenet_lite_416_coco.h5")

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

    cv2.imshow('YOLOv3 MobileNet', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()