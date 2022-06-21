import cv2
import numpy as np

import what.utils.logger as log
from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.yolov3 import YOLOV3
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors

from what.attacks.detection.yolo.TOG import TOGAttack

logger = log.get_logger(__name__)

def bilinear_resize_vectorized(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  img_height, img_width = image.shape

  image = image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_l = np.floor(x_ratio * x).astype('int32')
  y_l = np.floor(y_ratio * y).astype('int32')

  x_h = np.ceil(x_ratio * x).astype('int32')
  y_h = np.ceil(y_ratio * y).astype('int32')

  x_weight = (x_ratio * x) - x_l
  y_weight = (y_ratio * y) - y_l

  a = image[y_l * img_width + x_l]
  b = image[y_l * img_width + x_h]
  c = image[y_h * img_width + x_l]
  d = image[y_h * img_width + x_h]

  resized = a * (1 - x_weight) * (1 - y_weight) + \
            b * x_weight * (1 - y_weight) + \
            c * y_weight * (1 - x_weight) + \
            d * x_weight * y_weight

  return resized.reshape(height, width)

if __name__ == '__main__':
    # Read class names
    with open("examples/models/coco_classes.txt") as f:
        content = f.readlines()
    classes = [x.strip() for x in content] 

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    origin_cv_image = cv2.imread('examples/demo.png')
    origin_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_BGR2RGB)

    attack = TOGAttack("examples/models/yolov3.h5", "multi_untargeted", False, classes)
    attack.fixed = False

    for n in range(50):
        print("Iteration:", n)

        # For YOLO, the input pixel values are normalized to [0, 1]
        input_cv_image = cv2.resize(origin_cv_image, (416, 416))

        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        # Yolo inference
        # outs = model.predict(np.array([input_cv_image]))
        input_cv_image, outs = attack.attack(input_cv_image)

        boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))

        # (x, y, w, h) --> (x1, y1, x2, y2)
        height, width, _ = origin_cv_image.shape

        # Draw bounding boxes
        out_img = cv2.cvtColor(origin_cv_image, cv2.COLOR_RGB2BGR)
        out_img = out_img.astype(np.float32) / 255.0
        height, width, _ = out_img.shape
        noise = attack.noise
        noise_r = bilinear_resize_vectorized(noise[:, :, 0], height, width)
        noise_g = bilinear_resize_vectorized(noise[:, :, 1], height, width)
        noise_b = bilinear_resize_vectorized(noise[:, :, 2], height, width)
        noise = np.dstack((noise_r, noise_g, noise_b))

        out_img = out_img + noise
        out_img = np.clip(out_img, 0, 1)

        # input_cv_image = cv2.resize(input_cv_image, (width, height), interpolation = cv2.INTER_AREA)
        out_img = (out_img * 255.0).astype(np.uint8)

        for i in range(boxes.shape[0]):
            print(f"{classes[labels[i]]}: {probs[i]:.2f}")

        out_img = draw_bounding_boxes(out_img, boxes, labels, classes, probs);

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", out_img)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    print("Perturbation saved to noise.npy")
    np.save('noise.npy', attack.noise)
