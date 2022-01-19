import cv2
import torch
import numpy as np

from what.models.detection.frcnn.faster_rcnn import FasterRCNN
# from what.models.detection.frcnn.datasets.util import read_image

from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.datasets.voc import VOC_CLASS_NAMES

# Capture from camera
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

while True:
    _, orig_image = cap.read()
    if orig_image is None:
        continue

    # Image preprocessing
    input = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    # NHWC -> NCHW
    input = np.array(input).transpose((2, 0, 1))
    input = torch.from_numpy(input)[None]

    # img = read_image('notebooks/demo.jpg', format='NCHW')
    # input = torch.from_numpy(img)[None]
    # RGB --> BGR
    # img = img.transpose((1, 2, 0)).astype(np.uint8)

    model = FasterRCNN(device=device)
    model.load('models/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth', map_location=device)

    inputs, boxes, labels, scores = model.predict(input)

    output = draw_bounding_boxes(orig_image,
            boxes[0],
            labels[0],
            VOC_CLASS_NAMES[1:],
            scores[0])

    cv2.imshow('Faster RCNN', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
