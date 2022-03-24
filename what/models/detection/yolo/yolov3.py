import cv2
import numpy as np
from keras.models import load_model

from what.models.detection.utils.time_utils import Timer

from .utils.yolo_utils import yolo_process_output, yolov3_anchors

class YOLOV3:
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

        boxes, class_ids, confidences = yolo_process_output(outs, yolov3_anchors, len(self.class_names))

        return input_cv_image, boxes, class_ids, confidences
