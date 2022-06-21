import cv2
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.yolov3 import YOLOV3
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors

from what.attacks.detection.yolo.TOG import TOGAttack
from what.utils.resize import bilinear_resize
import what.utils.logger as log

logger = log.get_logger(__name__)

if __name__ == '__main__':
    # Read class names
    with open("models/coco_classes.txt") as f:
        content = f.readlines()
    classes = [x.strip() for x in content] 

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    origin_cv_image = cv2.imread('demo.png')
    origin_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_BGR2RGB)

    attack = TOGAttack("models/yolov3.h5", "multi_untargeted", False, classes)
    attack.fixed = False

    for n in range(30):
        logger.info(f"Iteration: {n}")

        # For YOLO, the input pixel values are normalized to [0, 1]
        input_cv_image = cv2.resize(origin_cv_image, (416, 416))
        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        # Yolo inference
        input_cv_image, outs = attack.attack(input_cv_image)
        boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))

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

        for i in range(boxes.shape[0]):
            logger.info(f"{classes[labels[i]]}: {probs[i]:.2f}")

        out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", out_img)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    logger.info("Perturbation saved to noise.npy")
    np.save('noise.npy', attack.noise)
