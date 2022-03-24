import cv2

def draw_bounding_boxes(image, boxes, labels, class_names, probs):
    if len(boxes) > 0:
        assert(boxes.shape[1] == 4)

    # Draw bounding boxes and labels
    for i in range(boxes.shape[0]):
        box = boxes[i]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        # print(label)

        # Draw bounding boxes
        cv2.rectangle(image, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), (255, 255, 0), 4)

        # Draw labels
        cv2.putText(image, label,
                    (int(box[0]+20), int(box[1]+40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    return image
