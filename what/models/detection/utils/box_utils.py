import cv2
import numpy as np
from .array_utils import to_numpy

def draw_bounding_boxes(image, boxes, labels, class_names, probs):
    if len(boxes) > 0:
        assert(boxes.shape[1] == 4)
        boxes = to_numpy(boxes)

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

    # Draw bounding boxes and labels
    for i in range(boxes.shape[0]):
        box = boxes[i]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        # print(label)

        # Draw bounding boxes
        cv2.rectangle(image, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), (255, 255, 0), 4)

        # Draw labels
        cv2.putText(image, label,
                    (int(box[0]+20), int(box[1]+40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    return image
