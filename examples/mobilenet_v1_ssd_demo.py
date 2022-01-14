import cv2
import sys
from what.models.detection.ssd.mobilenet_v1_ssd import MobileNetV1SSD

# Capture from camera
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# Initialize the model
if len(sys.argv) == 2:
    model = MobileNetV1SSD(model_path=sys.argv[1], is_test=True)
else:
    model = MobileNetV1SSD(pretrained=True, is_test=True)

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
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{model.class_names[labels[i]]}: {probs[i]:.2f}"
        print(label)

        # Draw bounding boxes
        cv2.rectangle(image, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), (255, 255, 0), 4)

        # Draw labels
        cv2.putText(image, label,
                    (int(box[0]+20), int(box[1]+40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type

    cv2.imshow('MobileNetv1 SSD', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
