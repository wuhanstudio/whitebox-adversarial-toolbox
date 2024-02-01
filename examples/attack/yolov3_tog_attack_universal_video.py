import cv2
import datetime
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors

from what.attacks.detection.yolo.TOG import TOGAttack
from what.utils.resize import bilinear_resize
import what.utils.logger as log

from what.cli.model import *
from what.utils.file import get_file
from what.utils.logger import TensorBoardLogger

# Loggingc
logger = log.get_logger(__name__)

CARLA_VIDEO_INDEX = 2

# Tensorboard
tog_log_dir = f'logs/tog-universal/carla_{CARLA_VIDEO_INDEX:04d}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
tb = TensorBoardLogger(tog_log_dir)

n_iteration = 100

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
    # Read video frames
    input_video = []

    cap = cv2.VideoCapture(f"carla/{CARLA_VIDEO_INDEX:04d}.mp4")
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
    model_path = os.path.join(WHAT_MODEL_PATH, what_yolov3_model_list[index][WHAT_MODEL_FILE_INDEX])
    attack = TOGAttack(model_path, classes)
    attack.fixed = False

    logger.info(f"Train: {len(x_train)}")

    origin_outs = []

    for i in range(len(x_train)):
        origin_cv_image = cv2.cvtColor(x_train[i, :, :, :], cv2.COLOR_BGR2RGB)

        # For YOLO, the input pixel values are normalized to [0, 1]
        input_cv_image = cv2.resize(origin_cv_image, (416, 416))
        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        # Yolo inference
        outs = attack.sess.run(attack.model.output, feed_dict={attack.model.input:np.array([input_cv_image])})

        origin_outs.append(outs)

    for n in range(n_iteration):

        res_mean_list = []
        boxes_list = []

        for i in range(len(x_train)):
            logger.info(f"Iteration: {n}, Frame: {i}")

            origin_cv_image = cv2.cvtColor(x_train[i, :, :, :], cv2.COLOR_BGR2RGB)

            # For YOLO, the input pixel values are normalized to [0, 1]
            input_cv_image = cv2.resize(origin_cv_image, (416, 416))

            input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

            # Yolo inference
            input_cv_image, outs = attack.attack(input_cv_image)
            boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))

            boxes_list.append(len(boxes))

            res_list = []
            for out, origin_out in zip(outs, origin_outs[i]):
                out = out.reshape((-1, 5 + len(classes)))
                origin_out = origin_out.reshape((-1, 5 + len(classes)))

                res = np.mean(out[:, 4] - origin_out[:, 4])
                res_list.append(res)

            res_mean_list.append(np.mean(res_list))

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

            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", out_img)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break

        tb.log_scalar('mean confidence increase', np.mean(res_mean_list), n)
        tb.log_scalar('boxes', np.mean(boxes_list), n)

        if (n+1) == 1 or (n+1) == 5 or (n+1) % 10 == 0:
            logger.info(f"Perturbation saved to noise_{CARLA_VIDEO_INDEX:04d}_{n}.npy")
            np.save(f"noise/noise_tog_{CARLA_VIDEO_INDEX:04d}_{n}.npy", attack.noise)
