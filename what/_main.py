import click

import os.path
from pathlib import Path

from what.utils.file import get_file

MODEL_PATH =  os.path.join(Path.home(), '.what', 'models')

what_model_list = [
    ('YOLOv3      ( Darknet )', 'Object Detection', 'YOLOv3 pretrained on MS COCO dataset.', 'yolov3.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolov3.h5', 'e557c671de8c46ff7d83a9a9b9750bcb2958a7275638c931106ada5d6057a26d'),
    ('YOLOv3      (Mobilenet)', 'Object Detection', 'YOLOv3 pretrained on MS COCO dataset.', 'yolov3_mobilenet_lite_416_coco.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolov3_mobilenet_lite_416_coco.h5', '93eb7be204fca8dd8bafa5146b9795978c9ca6a0ba1d2e2c002f4ed0a1322436'),
    ('YOLOv3 Tiny ( Darknet )', 'Object Detection', 'YOLOv3 Tiny pretrained on MS COCO dataset.', 'yolov3-tiny.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolov3-tiny.h5', '344387880e8ff5e45a313c85f2eeeb2fa4f6d3511ed1e5611aaed438db1f3876'),
    ('YOLOv3 Tiny (MobileNet)', 'Object Detection', 'YOLOv3 Tiny pretrained on MS COCO dataset.', 'tiny_yolo3_mobilenet_lite_416_coco.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/tiny_yolo3_mobilenet_lite_416_coco.h5', 'e124316a47915936baa39a157aca58cac86813b2c9bf49646e26c24e80252000'),
    ('YOLOv4      ( Darknet )', 'Object Detection', 'YOLOv4 pretrained on MS COCO dataset.', 'yolov4.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolov4.h5', '54802f99cdbddcb0f31180a55d30be1a80d0b73edaa13e9152829318387512e4'),
    ('YOLOv4 Tiny ( Darknet )', 'Object Detection', 'YOLOv4 Tiny pretrained on MS COCO dataset.', 'yolov4-tiny.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolov4-tiny.h5', '3d6f1a3d0540dd3d807de79cb0ad3374b5cdf7495abef5f89505f1c0fcbf8911')
]

what_attack_list = [
    ('TOG Attack', 'Object Detection', 'Adversarial Objectness Gradient Attacks in Real-time Object Detection Systems.'),
    ('PCB Attack', 'Object Detection', 'A Man-in-the-Middle Hardware Attack against Object Detection.')
]

# Main CLI (what)
@click.group()
def main_cli():
    """The CLI tool for WHite-box Adversarial Toolbox (WHAT)."""
    pass

# what model
@click.group()
def model():
    """Manage Deep Learning Models"""
    pass

# what model list
@model.command('list')
def model_list():
    """List supported models"""
    max_len = max([len(x[0]) for x in what_model_list])
    for i, model in enumerate(what_model_list, start=1):
        if os.path.isfile(os.path.join(MODEL_PATH, model[3])):
            downloaded = 'x'
        else:
            downloaded = ' '
        print('[{}] {} : {:<{w}s}\t{}\t{}'.format(downloaded, i, model[0], model[1], model[2], w=max_len))

# what model download
@model.command('download')
def model_download():
    """Download pre-trained models"""
    max_len = max([len(x[0]) for x in what_model_list])
    for i, model in enumerate(what_model_list, start=1):
        if os.path.isfile(os.path.join(MODEL_PATH, model[3])):
            downloaded = 'x'
        else:
            downloaded = ' '
        print('[{}] {} : {:<{w}s}\t{}\t{}'.format(downloaded, i, model[0], model[1], model[2], w=max_len))

    index = input(f"Please input the model index: ")
    while not index.isdigit() or int(index) > len(what_model_list):
        index = input(f"Model [{index}] does not exist. Please try again: ")
    
    index = int(index) - 1
    get_file(what_model_list[index][3], MODEL_PATH, what_model_list[index][4], what_model_list[index][5])

# what model run
@model.command('run')
def model_run():
    """Run supported models"""
    pass

# what attack
@click.group()
def attack():
    """Manage Attacks"""
    pass

# what attack list
@attack.command('list')
def attack_list():
    """List supported Attacks"""
    max_len = max([len(x[0]) for x in what_attack_list])
    for i, attack in enumerate(what_attack_list, start=1):
        print('{} : {:<{w}s}\t{}'.format(i, attack[0], attack[1], w=max_len))

# what example
@click.group()
def example():  
    """Manage Examples"""
    pass

# what example list
@example.command('list')
def example_list():
    """List examples"""
    pass

# what exmaple run
@example.command('run')
def example_run():
    """Run examples"""
    pass

def main():
    main_cli.add_command(model)
    main_cli.add_command(attack)
    main_cli.add_command(example)

    model.add_command(model_list)
    model.add_command(model_download)
    model.add_command(model_run)

    attack.add_command(attack_list)

    example.add_command(example_list)
    example.add_command(example_run)

    return main_cli()

if __name__ == "__main__":

    main()
