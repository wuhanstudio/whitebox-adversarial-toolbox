import cv2
import random
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors

from what.models.detection.yolo.yolov3 import YOLOV3
from what.utils.resize import bilinear_resize
import what.utils.logger as log

from what.cli.model import *
from what.utils.file import get_file

logger = log.get_logger(__name__)

# Target Model
what_yolov3_model_list = what_model_list[0:4]

# Check what_model_list for all supported models
index = 3

# Download the model first if not exists
WHAT_YOLOV3_MODEL_FILE = what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_YOLOV3_MODEL_URL  = what_yolov3_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_YOLOV3_MODEL_HASH = what_yolov3_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV3_MODEL_FILE)):
    get_file(WHAT_YOLOV3_MODEL_FILE,
             WHAT_MODEL_PATH,
             WHAT_YOLOV3_MODEL_URL,
             WHAT_YOLOV3_MODEL_HASH)

VIDEO_FILE = 'carla/0019.mp4'
noise = np.load('noise/noise_pcb_0019_99.npy')

if __name__ == '__main__':
    # Read video frames
    input_video = []

    cap = cv2.VideoCapture(VIDEO_FILE)
    success, image = cap.read()
    while success:
        input_video.append(image)
        success,image = cap.read()
    cap.release()

    classes = COCO_CLASS_NAMES 

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    logger.info(f"Read {len(input_video)} images")

    # random.shuffle(input_video)

    x_train = np.array(input_video[:int(len(input_video))])

    # Adversarial Attack
    model = YOLOV3(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV3_MODEL_FILE))

    logger.info(f"Test: {len(x_train)}")

    origin_outs = []
    origin_boxes = []
    origin_probs = []

    for i in range(len(x_train)):
        origin_cv_image = cv2.cvtColor(x_train[i, :, :, :], cv2.COLOR_BGR2RGB)

        # For YOLO, the input pixel values are normalized to [0, 1]
        input_cv_image = cv2.resize(origin_cv_image, (416, 416))
        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        # Run inference
        outs = model.model.predict(np.array([input_cv_image]))
        boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))

        origin_outs.append(outs)
        origin_boxes.append(boxes)
        origin_probs.append(probs)

    height, width, _ = origin_cv_image.shape

    noise_r = bilinear_resize(noise[:, :, 0], height, width)
    noise_g = bilinear_resize(noise[:, :, 1], height, width)
    noise_b = bilinear_resize(noise[:, :, 2], height, width)
    noise_l = np.dstack((noise_r, noise_g, noise_b))

    res_mean_list = []
    boxes_list = []
    box_var_list = []

    for i in range(len(x_train)):
        logger.info(f"Frame: {i}")

        origin_cv_image = cv2.cvtColor(x_train[i, :, :, :], cv2.COLOR_BGR2RGB)

        # For YOLO, the input pixel values are normalized to [0, 1]
        input_cv_image = cv2.resize(origin_cv_image, (416, 416))
        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        input_cv_image = input_cv_image + noise
        input_cv_image = np.clip(input_cv_image, 0.0, 1.0)

        # Run inference
        outs = model.model.predict(np.array([input_cv_image]))
        boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))

        boxes_list.append(len(boxes))

        res_list = []
        for out, origin_out in zip(outs, origin_outs[i]):
            out = out.reshape((-1, 5 + len(classes)))
            origin_out = origin_out.reshape((-1, 5 + len(classes)))

            res = np.mean(out[:, 4] - origin_out[:, 4])
            res_list.append(res)

        res_mean_list.append(np.mean(res_list))

        # Eliminate the boxes with low confidence and overlaped boxes
        if boxes.size > 0 and origin_boxes[i].size > 0:
            indexes = cv2.dnn.NMSBoxes(np.vstack((boxes, origin_boxes[i])).tolist(), np.hstack((np.array(probs), np.array(origin_probs[i]))), 0.5, 0.4)
            indexes = indexes.flatten()
            box_var_list.append( (len(boxes) + len(origin_boxes[i]) - len(indexes)) / len(boxes) )
        elif boxes.size == 0 and origin_boxes[i].size == 0:
            # No bounding boxes, all consistent
            box_var_list.append(1.0)
        else:
           # Either one is empty, none consistent
            box_var_list.append(0.0)

        # Draw bounding boxes
        origin_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_RGB2BGR)
        origin_cv_image = origin_cv_image.astype(np.float32) / 255.0

        origin_cv_image = origin_cv_image + noise_l
        origin_cv_image = np.clip(origin_cv_image, 0, 1)

        origin_cv_image = (origin_cv_image * 255.0).astype(np.uint8)
        out_img = draw_bounding_boxes(origin_cv_image, boxes, labels, classes, probs);

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", out_img)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    logger.info(f"mean confidence increase: {np.mean(res_mean_list)}")
    logger.info(f"boxes: {np.mean(boxes_list)}")
    logger.info(f"relative box variations: {np.mean(box_var_list)*100:.2f}%")
