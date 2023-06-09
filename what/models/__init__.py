r'''
This module implements several object detection models.


Use `what model list` to list available models:

```
                Model                      Model Type           Description
----------------------------------------------------------------------------------------------------
[ ] 1 : YOLOv3      (    Darknet    )   Object Detection        YOLOv3 pretrained on MS COCO dataset.
[ ] 2 : YOLOv3      (   Mobilenet   )   Object Detection        YOLOv3 pretrained on MS COCO dataset.
[ ] 3 : YOLOv3 Tiny (    Darknet    )   Object Detection        YOLOv3 Tiny pretrained on MS COCO dataset.
[ ] 4 : YOLOv3 Tiny (   MobileNet   )   Object Detection        YOLOv3 Tiny pretrained on MS COCO dataset.
[ ] 5 : YOLOv4      (    Darknet    )   Object Detection        YOLOv4 pretrained on MS COCO dataset.
[ ] 6 : YOLOv4 Tiny (    Darknet    )   Object Detection        YOLOv4 Tiny pretrained on MS COCO dataset.
[ ] 7 : SSD         ( MobileNet  v1 )   Object Detection        SSD pretrained on VOC-2012 dataset.
[ ] 8 : SSD         ( MobileNet  v2 )   Object Detection        SSD pretrained on VOC-2012 dataset.
[ ] 9 : FasterRCNN  (     VGG16     )   Object Detection        Faster-RCNN pretrained on VOC-2012 dataset.
```

## what.models.detection

'''

from what.models import detection
