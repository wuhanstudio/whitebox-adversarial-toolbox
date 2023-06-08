import cv2
import torch
import numpy as np

from what.models.detection.frcnn.faster_rcnn import FasterRCNN
# from what.models.detection.frcnn.datasets.util import read_image

from what.cli.model import *
from what.utils.file import get_file

from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.datasets.voc import VOC_CLASS_NAMES

# Download the model first if not exists
# Check what_model_list for all available models
index = 8
if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, what_model_list[index][WHAT_MODEL_FILE_INDEX])):
    get_file(what_model_list[index][WHAT_MODEL_FILE_INDEX],
                WHAT_MODEL_PATH,
                what_model_list[index][WHAT_MODEL_URL_INDEX],
                what_model_list[index][WHAT_MODEL_HASH_INDEX])

video = input(f"Please input the OpenCV capture device (e.g. 0, 1, 2): ")

while not video.isdigit():
    video = input(f"Please input the OpenCV capture device (e.g. 0, 1, 2): ")

# Capture from camera
cap = cv2.VideoCapture(int(video))
#cap.set(3, 1920)
#cap.set(4, 1080)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FasterRCNN(device=device)
model.load(os.path.join(WHAT_MODEL_PATH, what_model_list[8][WHAT_MODEL_FILE_INDEX]), map_location=device)

while True:
    _, orig_image = cap.read()
    if orig_image is None:
        continue

    # Image preprocessing
    input = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    height, width, _ = input.shape

    # NHWC -> NCHW
    input = np.array(input).transpose((2, 0, 1))
    input = torch.from_numpy(input)[None]

    # img = read_image('notebooks/demo.jpg', format='NCHW')
    # input = torch.from_numpy(img)[None]
    # RGB --> BGR
    # img = img.transpose((1, 2, 0)).astype(np.uint8)

    inputs, boxes, labels, scores = model.predict(input)

    # (x1, y1, x2, y2) --> (c1, c2, w, h) (0.0, 1.0)
    boxes = np.array(boxes)[0]
    box_w  = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] += box_w / 2
    boxes[:, 0] /= width
    boxes[:, 1] += box_h / 2
    boxes[:, 1] /= height
    boxes[:, 2] = box_w / width
    boxes[:, 3] = box_h / height

    output = draw_bounding_boxes(orig_image,
            boxes,
            labels[0],
            VOC_CLASS_NAMES[1:],
            scores[0])

    cv2.imshow('Faster RCNN', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
