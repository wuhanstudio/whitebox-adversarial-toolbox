import cv2
import os.path

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.yolo.yolov4 import YOLOV4

from what.cli.model import *

from what.utils.file import get_file

def yolov4_inference_demo():

    max_len = max([len(x[WHAT_MODEL_NAME_INDEX]) for x in what_model_list])
    for i, model in enumerate(what_model_list[0:2], start=1):
        if os.path.isfile(os.path.join(WHAT_MODEL_PATH, model[WHAT_MODEL_FILE_INDEX])):
            downloaded = 'x'
        else:
            downloaded = ' '
        print('[{}] {} : {:<{w}s}\t{}\t{}'.format(downloaded, i, model[0], model[1], model[2], w=max_len))

    index = input(f"Please input the model index: ")
    while not index.isdigit() or int(index) > len(what_model_list):
        index = input(f"Model [{index}] does not exist. Please try again: ")

    index = int(index) - 1

    # Download the model first if not exists
    if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, what_model_list[index][WHAT_MODEL_FILE_INDEX])):
        get_file(what_model_list[index][WHAT_MODEL_FILE_INDEX],
                 WHAT_MODEL_PATH,
                 what_model_list[index][WHAT_MODEL_URL_INDEX],
                 what_model_list[index][WHAT_MODEL_HASH_INDEX])

    video = input(f"Please input the OpenCV capture device (e.g. demo.mp4, demo.jpg, 0, 1, 2): ")

    if video.endswith('mp4') or video.isdigit():
        # Capture from camera or video
        if video.isdigit():
            cap = cv2.VideoCapture(int(video))
        else:
            cap = cv2.VideoCapture(video)

        #cap.set(3, 1920)
        #cap.set(4, 1080)

        model = YOLOV4(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, what_model_list[index][WHAT_MODEL_FILE_INDEX]))

        while True:
            _, orig_image = cap.read()
            if orig_image is None:
                continue

            # Image preprocessing
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

            # Run inference
            images, boxes, labels, probs = model.predict(image, 10, 0.4)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw bounding boxes onto the image
            output = draw_bounding_boxes(image, boxes, labels, model.class_names, probs);

            cv2.imshow('YOLOv3 MobileNet', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif video.endswith('jpg') or video.endswith('png'):
        orig_image = cv2.imread(video)

        model = YOLOV4(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, what_model_list[index][WHAT_MODEL_FILE_INDEX]))

        # Image preprocessing
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        # Run inference
        images, boxes, labels, probs = model.predict(image, 10, 0.4)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw bounding boxes onto the image
        output = draw_bounding_boxes(image, boxes, labels, model.class_names, probs);

        cv2.imshow('YOLOv4 Darknet', image)
        cv2.waitKey(0)

    else:
        print('Unsupported Capture Device "{}"'.format(video))
