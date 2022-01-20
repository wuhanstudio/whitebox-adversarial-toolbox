import torch
import cv2
import numpy as np

class FiftyOneDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for 
            training or testing
        transform (None): a list of PyTorch transform to apply to images 
            and targets when loading
        ground_truth_field ("ground_truth"): the name of the field in fiftyone_dataset 
            that contains the desired labels to load
        classes (None): a list of class strings that are used to define the 
            mapping between class names and indices. If None, it will use 
            all classes present in the given fiftyone_dataset.
    """

    def __init__(self, fiftyone_dataset, classes,
                 transform = None, target_transform = None,
                 ground_truth_field = "ground_truth"):

        self.dataset = fiftyone_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.ground_truth_field = ground_truth_field

        self.img_paths = self.dataset.values("filepath")

        self.classes = classes

        if self.classes[0] not in ["BACKGROUND", "background"]:
            self.classes = ["BACKGROUND"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.dataset[img_path]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        boxes = []
        labels = []
        detections = sample[self.ground_truth_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            x, y, w, h = det.bounding_box
            boxes.append(np.array([float(x)*width, float(y)*height, float(x + w)*width, float(y + h)*height]))
            labels.append(category_id)
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Background
        if len(boxes) == 0:
            boxes = np.array([[0, 0, img.shape[1], img.shape[0]]], dtype=np.float32)
            labels = np.array([0], dtype=np.int64)
    
        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
                
        return img, boxes, labels

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes
