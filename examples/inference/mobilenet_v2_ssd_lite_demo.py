import cv2
import torch

from what.cli.model import *
from what.utils.file import get_file

from what.models.detection.ssd.mobilenet_v2_ssd_lite import MobileNetV2SSDLite
from what.models.detection.utils.box_utils import draw_bounding_boxes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

video = input(f"Please input the OpenCV capture device (e.g. 0, 1, 2): ")

while not video.isdigit():
    video = input(f"Please input the OpenCV capture device (e.g. 0, 1, 2): ")

# Capture from camera
cap = cv2.VideoCapture(int(video))
#cap.set(3, 1920)
#cap.set(4, 1080)

# Download the model first if not exists
# Check what_model_list for all available models
index = 7
if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, what_model_list[index][WHAT_MODEL_FILE_INDEX])):
    get_file(what_model_list[index][WHAT_MODEL_FILE_INDEX],
                WHAT_MODEL_PATH,
                what_model_list[index][WHAT_MODEL_URL_INDEX],
                what_model_list[index][WHAT_MODEL_HASH_INDEX])

# Initialize the model
model = MobileNetV2SSDLite(os.path.join(WHAT_MODEL_PATH, what_model_list[index][WHAT_MODEL_FILE_INDEX]),
                           is_test=True,
                           device=device)

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

    cv2.imshow('MobileNetv2 SSD Lite', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
