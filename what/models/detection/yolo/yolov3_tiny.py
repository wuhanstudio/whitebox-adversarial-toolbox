import cv2
import numpy as np
from keras.models import load_model

from what.models.detection.utils.time_utils import Timer

from .utils.yolo_utils import yolo_process_output, yolov3_tiny_anchors

class YOLOV3_TINY:
    def __init__(self, class_names, model_path):
        self.model = load_model(model_path)
        self.class_names = class_names
        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        input_cv_image = cv2.resize(image, (416, 416))
        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        # Yolo inference
        self.timer.start()
        outs = self.model.predict(np.array([input_cv_image]))
        print("FPS: ", int(1.0 / self.timer.end()))

        boxes, class_ids, confidences = yolo_process_output(outs, yolov3_tiny_anchors, len(self.class_names))

        # (x, y, w, h) --> (x1, y1, x2, y2)
        height, width, _ = image.shape
        for box in boxes:
            box[0] *= width
            box[1] *= height
            box[2] *= width 
            box[3] *= height

            # From center to top left
            box[0] -= box[2] / 2
            box[1] -= box[3] / 2

            # From width and height to x2 and y2
            box[2] += box[0]
            box[3] += box[1]

        return input_cv_image, boxes, class_ids, confidences
