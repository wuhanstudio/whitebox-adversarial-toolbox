import cv2
import numpy as np

from keras.models import load_model
import tensorflow.keras.backend as K

from what.models.detection.utils.time_utils import Timer

from .utils.yolo_utils import yolo_process_output, yolov4_anchors

def mish(x):
    return x * K.tanh(K.softplus(x))

class YOLOV4:
    def __init__(self, class_names, model_path):
        self.model = load_model(model_path, custom_objects = {
            'mish': mish
        })
        self.class_names = class_names
        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        input_cv_image = cv2.resize(image, (416, 416))
        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        # Yolo inference
        self.timer.start()
        outs = self.model.predict(np.array([input_cv_image]))
        print("FPS: ", int(1.0 / self.timer.end()))

        boxes, class_ids, confidences = yolo_process_output(outs, yolov4_anchors, len(self.class_names))

        return input_cv_image, boxes, class_ids, confidences
