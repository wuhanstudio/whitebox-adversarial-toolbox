import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import fiftyone.zoo as foz

import what.utils.logger as log
from what.models.detection.frcnn.utils.config import opt

# from what.models.detection.frcnn.data.dataset import Dataset, TestDataset

from what.models.detection.frcnn.model.faster_rcnn_vgg16 import FasterRCNNVGG16
from what.models.detection.datasets.fiftyone import FiftyOneDataset

from torch.utils import data as data_

from what.models.detection.frcnn.faster_rcnn import FasterRCNN
from what.models.detection.utils.array_utils import to_scalar
# from what.models.detection.frcnn.utils.vis_tool import visdom_bbox

from what.models.detection.ssd.ssd.ssd import MatchPrior
from what.models.detection.ssd.ssd.preprocessing import TrainAugmentation, TestTransform
from what.models.detection.ssd.ssd import mobilenet_ssd_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = log.get_logger(__name__)

batch_size = 1
num_workers = 0

if __name__ == '__main__':

    # Load MobileNetSSD configuration
    config              = mobilenet_ssd_config
    train_transform     = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform    = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform      = TestTransform(config.image_size, config.image_mean, config.image_std)

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

    logger.info(f"Train dataset size: {len(train_dataset)}")

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

    logger.info(f"validation dataset size: {len(val_dataset)}")

    # dataset = Dataset(opt)
    # dataloader = data_.DataLoader(dataset, \
    #                               batch_size=1, \
    #                               shuffle=True, \
    #                               # pin_memory=True,
    #                               num_workers=opt.num_workers)
    # testset = TestDataset(opt)
    # test_dataloader = data_.DataLoader(testset,
    #                                    batch_size=1,
    #                                    num_workers=2,
    #                                    shuffle=False, \
    #                                    # pin_memory=True
    #                                    )

    model = FasterRCNN()

    if opt.load_path:
        model.load(opt.load_path)
        logger.info(f'load pretrained model from {opt.load_path}')

    # trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    for epoch in range(7):
        model.reset_meters()
        for ii, (img, bbox_, label_) in tqdm(enumerate(train_loader)):
            scale = 300 / img.shape[0]
            img, bbox, label = img.to(device).float(), bbox_.to(device), label_.to(device)
            losses = model.step(img, bbox, label, scale)

    eval_result = model.eval(val_loader, test_num=1e100)
    model.save(mAP=eval_result['map'])

    logger.info('eval_result')
