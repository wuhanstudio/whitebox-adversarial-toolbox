import cv2
import datetime
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors, yolov3_tiny_anchors

from what.attacks.detection.yolo.PCB import PCBAttack
from what.utils.resize import bilinear_resize
import what.utils.logger as log

from what.utils.logger import TensorBoardLogger

import fiftyone.zoo as foz

from what.cli.model import *
from what.utils.file import get_file

n_iteration = 50
prefix = './'

show_image = False

# Logging
logger = log.get_logger(__name__)

# Tensorboard
pcb_log_dir = prefix + 'logs/pcb-universal/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoardLogger(pcb_log_dir)

# Target Model
what_yolov3_model_list = what_model_list[0:4]

if __name__ == '__main__':

    # Load Training Dataset from FiftyOne
    train_dataset = foz.load_zoo_dataset("voc-2012", split="validation")
    img_paths = train_dataset.values("filepath")

    classes = COCO_CLASS_NAMES

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

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
    attack = PCBAttack(model_path, "multi_untargeted", classes, learning_rate=0.001, batch=len(img_paths))
    attack.fixed = False

    origin_outs = []

    for i in range(len(img_paths)):
        img_path = img_paths[i]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # For YOLO, the input pixel values are normalized to [0, 1]
        input_cv_image = cv2.resize(img, (416, 416))
        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        # Yolo inference
        outs = attack.sess.run(attack.model.output, feed_dict={attack.model.input:np.array([input_cv_image])})

        origin_outs.append(outs)

    for n in range(0, n_iteration):

        logger.info(f"Iteration: {n}")

        res_mean_list = []
        boxes_list = []

        for i in range(len(img_paths)):
            img_path = img_paths[i]

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # For YOLO, the input pixel values are normalized to [0, 1]
            input_cv_image = cv2.resize(img, (416, 416))
            input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

            # Yolo inference
            input_cv_image, outs = attack.attack(input_cv_image)

            boxes, labels, probs = yolo_process_output(outs, yolov3_tiny_anchors, len(classes))

            boxes_list.append(len(boxes))

            # Draw bounding boxes
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out_img = out_img.astype(np.float32) / 255.0
            height, width, _ = out_img.shape

            # noise = attack.noise
            # noise_r = bilinear_resize(noise[:, :, 0], height, width)
            # noise_g = bilinear_resize(noise[:, :, 1], height, width)
            # noise_b = bilinear_resize(noise[:, :, 2], height, width)
            # noise = np.dstack((noise_r, noise_g, noise_b))

            # out_img = out_img + noise

            res_list = []
            for out, origin_out in zip(outs, origin_outs[i]):
                out = out.reshape((-1, 5 + len(classes)))
                origin_out = origin_out.reshape((-1, 5 + len(classes)))

                res = np.mean(out[:, 4] - origin_out[:, 4])
                res_list.append(res)

            res_mean_list.append(np.mean(res_list))

            if show_image:
                out_img = np.clip(out_img, 0, 1)

                out_img = (out_img * 255.0).astype(np.uint8)

                for i in range(boxes.shape[0]):
                    logger.info(f"{classes[labels[i]]}: {probs[i]:.2f}")

                out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);

                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", out_img)

                cv2.waitKey(1)
                if (cv2.waitKey(0) & 0xFF == ord('q')):
                    break

        tb.log_scalar('mean confidence increase', np.mean(res_mean_list), n)
        tb.log_scalar('boxes', np.mean(boxes_list), n)

        logger.info("Perturbation saved to noise.npy")
        np.save(prefix + 'noise/noise-' + str(n) + '.npy', attack.noise)
