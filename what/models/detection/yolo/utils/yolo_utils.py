import cv2
import numpy as np
from scipy.special import expit

YOLOV3_INPUT_SIZE = 416

yolov3_anchors = [ 
            [[116,90],  [156,198],  [373,326]], 
            [[30,61],  [62,45],  [59,119]],  
            [[10,13],  [16,30],  [33,23]] 
            ]

yolov3_tiny_anchors = [ 
            [[81,82],  [135,169],  [344,319]],
            [[10,14],  [23,27],  [37,58]]
            ]

yolov4_anchors = [
            [[142,110], [192,243], [459,401]],
            [[36,75], [76,55], [72,146]],
            [[12,16], [19,36], [40,28]]
            ]

yolov4_tiny_anchors = [ 
            [[81,82],  [135,169],  [344,319]],
            [[10,14],  [23,27],  [37,58]]
            ]

def yolo_process_output(outs, anchors, num_classes):
    # The output of YOLO consists of three scales, each scale has three anchor boxes

    # For different scales, we need to calculate the actual size of three potential anchor boxes
    class_ids = []
    confidences = []
    boxes = []

    for anchor_i, out in enumerate(outs):

        anchor = anchors[anchor_i]
        num_anchors = len(anchor)

        grid_size = np.shape(out)[1:3]
        out = out.reshape((-1, 5 + num_classes))

        # The output of each bounding box is relative to the grid
        # Thus we need to add an offset to retrieve absolute coordinates
        grid_y = np.arange(grid_size[0])
        grid_x = np.arange(grid_size[1])
        x_offset, y_offset = np.meshgrid(grid_x, grid_y)

        x_offset = np.reshape(x_offset, (-1, 1))
        y_offset = np.reshape(y_offset, (-1, 1))

        x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
        x_y_offset = np.tile(x_y_offset, (1, num_anchors))
        x_y_offset = np.reshape(x_y_offset, (-1, 2))

        anchor = np.tile(anchor, (grid_size[0] * grid_size[1], 1))

        # The output format of each bounding box is (x, y, w, h)
        box_xy = (expit(out[..., :2]) + x_y_offset) / np.array(grid_size)[::-1]
        box_wh = (np.exp(out[..., 2:4]) * anchor) / np.array((YOLOV3_INPUT_SIZE, YOLOV3_INPUT_SIZE))[::-1]

        # Calculate the confidence value of each bounding box
        score = expit(out[:, 5:]) # class_scores
        class_id = np.argmax(score, axis=1)
        confidence = np.expand_dims(expit(out[:, 4]), -1) * score # class_score * objectness
        confidence = np.max(confidence, axis=-1)

        # We are only interested with the box with high confidence
        confidence_threshold = 0.1
        box_xy = box_xy[confidence > confidence_threshold]
        box_wh = box_wh[confidence > confidence_threshold]
        class_id = class_id[confidence > confidence_threshold]
        score = score[confidence > confidence_threshold]
        confidence = confidence[confidence > confidence_threshold]

        if(len(confidence) > 0):
            box_tmp = list(np.concatenate((box_xy, box_wh), axis=1))
            for b in box_tmp:
                boxes.append(b)
            for c in confidence:
                confidences.append(float(c))
            for c in class_id:
                class_ids.append(c)

    # Eliminate the boxes with low confidence and overlaped boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        indexes = indexes.flatten()

    boxes = [boxes[i] for i in indexes]
    boxes = np.array(boxes)

    confidences = [confidences[i] for i in indexes]
    class_ids = [class_ids[i] for i in indexes]

    return boxes, class_ids, confidences
