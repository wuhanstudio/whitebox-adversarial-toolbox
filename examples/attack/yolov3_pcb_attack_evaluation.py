import cv2
import time
import datetime
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors, yolov3_tiny_anchors

from what.attacks.detection.yolo.PCB import PCBAttack
from what.utils.resize import bilinear_resize
import what.utils.logger as log

from what.utils.logger import TensorBoardLogger

from what.cli.model import *
from what.utils.file import get_file

n_iteration = 500

prefix = './'

# Logging
logger = log.get_logger(__name__)

# Tensorboard
pcb_log_dir = prefix + 'logs/pcb/uniform_init/decay/0.98/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# pcb_log_dir = prefix + 'logs/pcb/zero_init/decay/0.98/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoardLogger(pcb_log_dir)

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

if __name__ == '__main__':

    classes = COCO_CLASS_NAMES

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    origin_cv_image = cv2.imread(prefix + 'demo.png')
    origin_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_BGR2RGB)

    # Adversarial Attack
    model_path = os.path.join(WHAT_MODEL_PATH, what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX])
    attack = PCBAttack(model_path, "multi_untargeted", classes, init="uniform", decay=0.98)
    attack.fixed = False

    last_outs = None
    last_boxes = None
    last_probs = None

    attack_time = []
    for n in range(0, n_iteration):
        logger.info(f"Iteration: {n}")

        # For YOLO, the input pixel values are normalized to [0, 1]
        input_cv_image = cv2.resize(origin_cv_image, (416, 416))
        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        # Yolo inference
        start_time = time.time()
        input_cv_image, outs = attack.attack(input_cv_image)
        attack_time.append(time.time() - start_time)

        tb.log_scalar('attack time', 1.0 / attack_time[-1], n)
        print("FPS:", 1.0 / attack_time[-1])

        if last_outs is not None:
            res_list = []
            for out, last_out in zip(outs, last_outs):
                out = out.reshape((-1, 5 + len(classes)))
                last_out = last_out.reshape((-1, 5 + len(classes)))

                res = np.mean(out[:, 4] - last_out[:, 4])
                res_list.append(res)
                logger.info(f"Increased: {res}")

            tb.log_scalar('mean confidence increase', np.mean(res_list), n)
        else:
            tb.log_scalar('mean confidence increase', 0.0, n)
            last_outs = outs

        boxes, labels, probs = yolo_process_output(outs, yolov3_tiny_anchors, len(classes))

        tb.log_scalar('boxes', len(boxes), n)

        if last_boxes is not None:
            # Eliminate the boxes with low confidence and overlaped boxes
            if last_boxes.size > 0 and boxes.size > 0:
                indexes = cv2.dnn.NMSBoxes(np.vstack((boxes, last_boxes)).tolist(), np.hstack((np.array(probs), np.array(last_probs))), 0.5, 0.4)
                if len(indexes) > 0:
                    indexes = indexes.flatten()
                    tb.log_scalar('box variation', (len(boxes) + len(last_boxes) - len(indexes)) / len(boxes), n)
            elif last_boxes.size == 0 and boxes.size == 0:
                # No bounding boxes, all consistent
                tb.log_scalar('box variation', 1.0, n)
            else:
                # Either one is empty, none consistent
                tb.log_scalar('box variation', 0.0, n)
        else:
            tb.log_scalar('box variation', 1.0, n)

        last_boxes = np.copy(boxes)
        last_probs = np.copy(probs)

        # Draw bounding boxes
        out_img = cv2.cvtColor(origin_cv_image, cv2.COLOR_RGB2BGR)
        out_img = out_img.astype(np.float32) / 255.0
        height, width, _ = out_img.shape
        noise = attack.noise
        noise_r = bilinear_resize(noise[:, :, 0], height, width)
        noise_g = bilinear_resize(noise[:, :, 1], height, width)
        noise_b = bilinear_resize(noise[:, :, 2], height, width)
        noise = np.dstack((noise_r, noise_g, noise_b))

        out_img = out_img + noise
        out_img = np.clip(out_img, 0, 1)

        out_img = (out_img * 255.0).astype(np.uint8)

        # for i in range(boxes.shape[0]):
        #     logger.info(f"{classes[labels[i]]}: {probs[i]:.2f}")

        out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", out_img)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    
    print("Average FPS:", 1.0 / np.mean(attack_time))
    logger.info("Perturbation saved to noise.npy")
    np.save(prefix + 'noise/noise.npy', attack.noise)
