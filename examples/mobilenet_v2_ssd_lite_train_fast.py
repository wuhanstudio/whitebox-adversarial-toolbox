import fiftyone.zoo as foz
import what.utils.logger as log

import torch
from torch.utils.data import DataLoader

from what.models.detection.datasets.fiftyone import FiftyOneDataset
from what.models.detection.datasets.voc import VOCDataset

from what.models.detection.ssd.ssd.ssd import MatchPrior
from what.models.detection.ssd.ssd.preprocessing import TrainAugmentation, TestTransform
from what.models.detection.ssd.ssd import mobilenet_ssd_config

from what.models.detection.ssd.mobilenet_v2_ssd_lite import MobileNetV2SSDLite

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

logger = log.get_logger(__name__)

if __name__ == '__main__':

    pretrained_ssd = None                   # or use "models/mobilenet-v2-ssd-lite-mp-0_686.pth"
    checkpoint_folder = "checkpoint/"

    batch_size = 32
    num_workers = 0

    # Load MobileNetSSD configuration
    config              = mobilenet_ssd_config
    train_transform     = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform    = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform      = TestTransform(config.image_size, config.image_mean, config.image_std)

    # Visulize the VOC2012 dataset
    # session = fo.launch_app(foz.load_zoo_dataset("voc-2012", split="train"))
    # session.wait()

    # Load Training Dataset from FiftyOne
    train_dataset = FiftyOneDataset(foz.load_zoo_dataset("voc-2012", split="train"), 
                                            foz.load_zoo_dataset_info("voc-2012").classes, 
                                            transform=train_transform,
                                            target_transform=target_transform)

    # Load Training Dataset from Local Disk
    # train_dataset = VOCDataset("VOC2012/", transform=train_transform,
    #                              target_transform=target_transform)

    train_loader  = DataLoader(train_dataset, batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    logger.info("Train dataset size: {}".format(len(train_dataset)))

    # Load Validation Dataset from FiftyOne (use voc-2007 train as validation here)
    val_dataset  = FiftyOneDataset(foz.load_zoo_dataset("voc-2007", split="train"), 
                                            foz.load_zoo_dataset_info("voc-2007").classes,
                                            transform=test_transform,
                                            target_transform=target_transform)

    # Load Validation Dataset from Local Disk
    # val_dataset = VOCDataset("VOC2007/", transform=test_transform,
    #                              target_transform=target_transform, is_test=True)

    val_loader = DataLoader(val_dataset, batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    logger.info("validation dataset size: {}".format(len(val_dataset)))

    # Create SSD network and load pretrained base net.
    model = MobileNetV2SSDLite(is_test=False, class_names=train_dataset.classes)

    model.train(train_loader, val_loader, device=device, num_epochs=5, debug_steps=10, validation_epochs=1,
                pretrained_ssd = pretrained_ssd, checkpoint_folder = checkpoint_folder)
