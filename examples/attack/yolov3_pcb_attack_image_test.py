import cv2
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors, yolov3_tiny_anchors

from what.utils.resize import bilinear_resize
from what.models.detection.yolo.yolov3_tiny import YOLOV3_TINY

from what.cli.model import *
from what.utils.file import get_file

# Target Model
what_yolov3_model_list = what_model_list[0:4]

noise = np.load('noise.npy')

if __name__ == '__main__':

    classes = COCO_CLASS_NAMES

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    origin_cv_image = cv2.imread('demo.jpg')
    origin_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_BGR2RGB)

    # Check what_model_list for all supported models
    index = 3

    # Download the model first if not exists
    if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX])):
        get_file(what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX],
                    WHAT_MODEL_PATH,
                    what_yolov3_model_list[index][WHAT_MODEL_URL_INDEX],
                    what_yolov3_model_list[index][WHAT_MODEL_HASH_INDEX])

    # Check what_model_list for all supported models
    model = YOLOV3_TINY(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX]))

    origin_cv_image = cv2.imread('demo.jpg')
    input_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_BGR2RGB)
    input_cv_image = cv2.resize(input_cv_image, (416, 416))

    input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0
    input_cv_image = input_cv_image + noise
    input_cv_image = np.clip(input_cv_image, 0.0, 1.0) * 255.0
    input_cv_image = input_cv_image.astype(np.uint8)

    # Run inference
    images, boxes, labels, probs = model.predict(input_cv_image, 10, 0.4)
    image = cv2.cvtColor(input_cv_image, cv2.COLOR_RGB2BGR)

    out_img = draw_bounding_boxes(image, boxes, labels, model.class_names, probs);

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", out_img)

    cv2.waitKey(0)
