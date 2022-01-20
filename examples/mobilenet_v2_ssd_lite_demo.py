import cv2
import sys
import torch
from what.models.detection.ssd.mobilenet_v2_ssd_lite import MobileNetV2SSDLite
from what.models.detection.utils.box_utils import draw_bounding_boxes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Capture from camera
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# Initialize the model
if len(sys.argv) == 2:
    model = MobileNetV2SSDLite(model_path=sys.argv[1], is_test=True, device=device)
else:
    model = MobileNetV2SSDLite(pretrained=True, is_test=True, device=device)

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
