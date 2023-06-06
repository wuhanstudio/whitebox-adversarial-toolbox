from what.examples.yolov3_demo import yolov3_inference_demo
from what.examples.yolov4_demo import yolov4_inference_demo
from what.examples.faster_rcnn_demo import frcnn_inference_demo

WHAT_EXAMPLE_NAME_INDEX = 0
WHAT_EXAMPLE_TYPE_INDEX = 1
WHAT_EXAMPLE_DESC_INDEX = 2
WHAT_EXAMPLE_FUNC_INDEX = 3

what_example_list = [
    ('  Yolov3 Demo  ', ' Model Inference ', 'Yolov3 Object Detection.', yolov3_inference_demo),
    ('  Yolov4 Demo  ', ' Model Inference ', 'Yolov4 Object Detection.', yolov4_inference_demo),
    ('FasterRCNN Demo', ' Model Inference ', ' FRCNN Object Detection.', frcnn_inference_demo),
]
