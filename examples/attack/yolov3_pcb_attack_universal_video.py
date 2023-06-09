import cv2
import random
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors

from what.attacks.detection.yolo.PCB import PCBAttack
from what.utils.resize import bilinear_resize
import what.utils.logger as log

from what.cli.model import *
from what.utils.file import get_file

logger = log.get_logger(__name__)

# Target Model
what_yolov3_model_list = what_model_list[0:4]

if __name__ == '__main__':
    # Read video frames
    input_video = []

    cap = cv2.VideoCapture('demo.mp4')
    success, image = cap.read()
    while success:
        input_video.append(image)
        success,image = cap.read()
    cap.release()

    classes = COCO_CLASS_NAMES 

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    logger.info(f"Read {len(input_video)} images")

    # random.shuffle(input_video)

    x_train = np.array(input_video[:int(len(input_video) * 0.9)])
    x_test = np.array(input_video[int(len(input_video) * 0.9):])

    # Check what_model_list for all supported models
    index = 3

    # Download the model first if not exists
    if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX])):
        get_file(what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX],
                    WHAT_MODEL_PATH,
                    what_yolov3_model_list[index][WHAT_MODEL_URL_INDEX],
                    what_yolov3_model_list[index][WHAT_MODEL_HASH_INDEX])

    # Adversarial Attack
    model_path = os.path.join(WHAT_MODEL_PATH, what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX])
    attack = PCBAttack(model_path, "multi_untargeted", classes)
    attack.fixed = False

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

    logger.info(f"Train: {len(x_train)}")
    logger.info(f"Test: {len(x_test)}")

    for n in range(5):
        for i in range(len(x_train)):
            logger.info(f"Iteration: {n}, Frame: {i}")

            origin_cv_image = cv2.cvtColor(x_train[i, :, :, :], cv2.COLOR_BGR2RGB)

            # For YOLO, the input pixel values are normalized to [0, 1]
            input_cv_image = cv2.resize(origin_cv_image, (416, 416))

            input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

            # Yolo inference
            input_cv_image, outs = attack.attack(input_cv_image)
            boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))

            # Draw bounding boxes
            origin_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_RGB2BGR)
            origin_cv_image = origin_cv_image.astype(np.float32) / 255.0
            height, width, _ = origin_cv_image.shape
            noise = attack.noise
            noise_r = bilinear_resize(noise[:, :, 0], height, width)
            noise_g = bilinear_resize(noise[:, :, 1], height, width)
            noise_b = bilinear_resize(noise[:, :, 2], height, width)
            noise = np.dstack((noise_r, noise_g, noise_b))

            origin_cv_image = origin_cv_image + noise
            origin_cv_image = np.clip(origin_cv_image, 0, 1)

            origin_cv_image = (origin_cv_image * 255.0).astype(np.uint8)
            out_img = draw_bounding_boxes(origin_cv_image, boxes, labels, classes, probs);

            out.write(out_img)

            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", out_img)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    out.release()

    logger.info("Perturbation saved to noise.npy")
    np.save('noise.npy', attack.noise)
