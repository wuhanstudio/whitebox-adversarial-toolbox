import cv2
import time
import random
import numpy as np

import what.utils.logger as log
from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.yolov3 import YOLOV3
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov3_anchors

from what.attacks.detection.yolo.CBP import CBPAttack

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
    # Read video frames
    input_video = []

    cap = cv2.VideoCapture('examples/demo.mp4')
    success, image = cap.read()
    while success:
        input_video.append(image)
        success,image = cap.read()
    cap.release()

    # Read class names
    with open("examples/models/coco_classes.txt") as f:
        content = f.readlines()
    classes = [x.strip() for x in content] 

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    print("Read {} images".format(len(input_video)))
    random.shuffle(input_video)
    x_train = np.array(input_video[:int(len(input_video) * 0.9)])
    x_test = np.array(input_video[int(len(input_video) * 0.9):])

    attack = CBPAttack("examples/models/yolov3.h5", "multi_untargeted", False, classes)
    attack.fixed = False

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

    print("Train: {}".format(len(x_train)))
    print("Test: {}".format(len(x_test)))

    for n in range(5):
        for i in range(len(x_train)):
            print("Iteration:", n, " Frame:", i)

            origin_cv_image = cv2.cvtColor(x_train[i, :, :, :], cv2.COLOR_BGR2RGB)

            # For YOLO, the input pixel values are normalized to [0, 1]
            input_cv_image = cv2.resize(origin_cv_image, (416, 416))

            input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

            start_time = int(time.time() * 1000)

            # Yolo inference
            # outs = model.predict(np.array([input_cv_image]))
            input_cv_image, outs = attack.attack(input_cv_image)

            boxes, labels, probs = yolo_process_output(outs, yolov3_anchors, len(classes))

            # (x, y, w, h) --> (x1, y1, x2, y2)
            height, width, _ = origin_cv_image.shape
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

            # Calculate FPS
            elapsed_time = int(time.time()*1000) - start_time
            fps = 1000 / elapsed_time
            print ("fps: ", str(round(fps, 2)))

            # Draw bounding boxes
            origin_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_RGB2BGR)
            origin_cv_image = origin_cv_image.astype(np.float32) / 255.0
            height, width, _ = origin_cv_image.shape
            noise = attack.noise
            noise_r = bilinear_resize_vectorized(noise[:, :, 0], height, width)
            noise_g = bilinear_resize_vectorized(noise[:, :, 1], height, width)
            noise_b = bilinear_resize_vectorized(noise[:, :, 2], height, width)
            noise = np.dstack((noise_r, noise_g, noise_b))

            origin_cv_image = origin_cv_image + noise
            origin_cv_image = np.clip(origin_cv_image, 0, 1)

            # input_cv_image = cv2.resize(input_cv_image, (width, height), interpolation = cv2.INTER_AREA)
            origin_cv_image = (origin_cv_image * 255.0).astype(np.uint8)
            out_img = draw_bounding_boxes(origin_cv_image, boxes, labels, classes, probs);

            out.write(out_img)

            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", out_img)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    out.release()
    np.save('examples/noise.npy', attack.noise)
