import cv2
import torch
import numpy as np

from keras.models import load_model

from .data.datasets import COCO_CLASSES
from .exp import get_exp
from .predictor import Predictor

class YOLOX_M:
    def __init__(self, class_names, model_path):
        exp = get_exp(None, "yolox-m")
        self.model = exp.get_model()

        # load the model state dict
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])

        self.predictor = Predictor(
                self.model, exp, COCO_CLASSES, None, None,
                device, False, False,
            )

        self.class_names = class_names

    def predict(self, image):
        outputs, _ = self.predictor.inference(image)
        height, width, _ = image.shape
        scale = min(640 / float(height), 640 / float(width))

        # Run inference
        class_ids = []
        boxes = []
        probs = []

        if outputs[0] is not None:
            boxes  = outputs[0][:, 0:4].cpu().numpy()
            class_ids = outputs[0][:, -1].cpu().numpy().astype(np.uint32)
            probs = (outputs[0][:, 4] * outputs[0][:, 5]).cpu().numpy()

            boxes = np.array([box for box, prob in zip(boxes, probs) if prob >= 0.5 ])
            probs = probs[probs >= 0.5]

        # From (x1, y1, x2, y2) --> (x, y, w, h)
        for i, box in enumerate(boxes):
            box /= scale
            box[0] /= width
            box[1] /= height
            box[2] /= width 
            box[3] /= height

            box[2] -= box[0]
            box[3] -= box[1]
            box[0] += (box[2] / 2)
            box[1] += (box[3] / 2)

        return image, boxes, class_ids, probs
