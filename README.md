<img src="https://what.wuhanstudio.uk/images/what.png" width=300px style="float: left;" >

# WHite-box Adversarial Toolbox (WHAT)

<!-- [![CircleCI](https://circleci.com/gh/wuhanstudio/whitebox-adversarial-toolbox.svg?style=svg)](https://circleci.com/gh/wuhanstudio/whitebox-adversarial-toolbox) -->
[![Build Status](https://app.travis-ci.com/wuhanstudio/whitebox-adversarial-toolbox.svg?branch=master)](https://app.travis-ci.com/wuhanstudio/whitebox-adversarial-toolbox)
[![PyPI version](https://badge.fury.io/py/whitebox-adversarial-toolbox.svg)](https://badge.fury.io/py/whitebox-adversarial-toolbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/whitebox-adversarial-toolbox)](https://pypi.org/project/whitebox-adversarial-toolbox/)
[![](https://img.shields.io/badge/Documentation-infromational)](https://what.wuhanstudio.uk/)

A Python Library for Deep Learning Security that focuses on Real-time White-box Attacks.

![](docs/images/demo.gif)

## Installation

```python
pip install whitebox-adversarial-toolbox
```

## Usage (CLI)

```
Usage: what [OPTIONS] COMMAND [ARGS]...

  The CLI tool for WHitebox-box Adversarial Toolbox (what).

Options:
  --help  Show this message and exit.

Commands:
  attack   Manage Attacks
  example  Manage Examples
  model    Manage Deep Learning Models
```

Useful commands:

```
# List supported models
$ what model list

# List supported Attacks
$ what attack list

# List available examples
$ what example list
```

Available models:

```
[x] 1 : YOLOv3      (    Darknet    )   Object Detection        YOLOv3 pretrained on MS COCO dataset.
[x] 2 : YOLOv3      (   Mobilenet   )   Object Detection        YOLOv3 pretrained on MS COCO dataset.
[x] 3 : YOLOv3 Tiny (    Darknet    )   Object Detection        YOLOv3 Tiny pretrained on MS COCO dataset.
[x] 4 : YOLOv3 Tiny (   MobileNet   )   Object Detection        YOLOv3 Tiny pretrained on MS COCO dataset.
[x] 5 : YOLOv4      (    Darknet    )   Object Detection        YOLOv4 pretrained on MS COCO dataset.
[x] 6 : YOLOv4 Tiny (    Darknet    )   Object Detection        YOLOv4 Tiny pretrained on MS COCO dataset.
[x] 7 : SSD         ( MobileNet  v1 )   Object Detection        SSD pretrained on VOC-2012 dataset.
[x] 8 : SSD         ( MobileNet  v2 )   Object Detection        SSD pretrained on VOC-2012 dataset.
[x] 9 : FasterRCNN  (     VGG16     )   Object Detection        Faster-RCNN pretrained on VOC-2012 dataset.
```

## A Man-in-the-Middle Hardware Attack

The Universal Adversarial Perturbation (UAP) can be deployed using a Man-in-the-Middle Hardware Attack.

[[ Talk ]](https://minm.wuhanstudio.uk) [[ Video ]](https://youtu.be/OvIpe-R3ZS8) [[ Paper ]](https://arxiv.org/abs/2208.07174) [[ Code ]](https://github.com/wuhanstudio/adversarial-camera)

![](https://github.com/wuhanstudio/adversarial-camera/raw/master/doc/demo.png)

![](https://github.com/wuhanstudio/adversarial-camera/raw/master/doc/demo.jpg)

The Man-in-the-Middle Attack consists of two steps:

- Step 1: [Generating the perturbation](detection/README.md).
- Step 2: [Deploying the perturbation](hardware/README.md).
