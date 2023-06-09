import os
import itertools

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import what.utils.logger as log

from .ssd import mobilenet_ssd_config
from .ssd.multibox_loss import MultiboxLoss
from .ssd.mobilenet_v2_ssd_lite_create import create_mobilenet_v2_ssd_lite, create_mobilenet_v2_ssd_lite_predictor

from .utils.misc import freeze_net_layers

logger = log.get_logger(__name__)

from what.models.detection.datasets.voc import VOC_CLASS_NAMES

class MobileNetV2SSDLite:
    def __init__(self, class_names = None, model_path = None, pretrained = None, width_mult = 1.0, is_test=False, device=None):

        if class_names is None:
            self.class_names = VOC_CLASS_NAMES
        else:
            self.class_names = class_names

        self.net = create_mobilenet_v2_ssd_lite(len(self.class_names), is_test=is_test, width_mult=width_mult)

        if model_path is not None:
            pretrained = False

        self.predictor = None;
        self.device = device;

        if pretrained is True:
            self.net.load("https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth", pretrained=True)
        elif model_path is not None:
            self.net.load(model_path)

    def predict(self, image, top_k=-1, prob_threshold=None):
        if self.predictor is None:
            self.predictor = create_mobilenet_v2_ssd_lite_predictor(self.net, candidate_size=200, device=self.device)

        return self.predictor.predict(image, top_k, prob_threshold)

    def step(self, loader, criterion, optimizer, device, debug_steps=100, epoch=-1):
        self.net.train(True)
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        for i, data in enumerate(loader):
            images, boxes, labels = data
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            confidence, locations = self.net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
            loss = regression_loss + classification_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
            if i and i % debug_steps == 0:
                avg_loss = running_loss / debug_steps
                avg_reg_loss = running_regression_loss / debug_steps
                avg_clf_loss = running_classification_loss / debug_steps
                logger.info(
                    f"Epoch: {epoch}, Step: {i}, " +
                    f"Average Loss: {avg_loss:.4f}, " +
                    f"Average Regression Loss {avg_reg_loss:.4f}, " +
                    f"Average Classification Loss: {avg_clf_loss:.4f}"
                )
                running_loss = 0.0
                running_regression_loss = 0.0
                running_classification_loss = 0.0

    def train(self, train_loader, val_loader, device = "cpu", 
            scheduler = None, criterion = None, optimizer = None, 
            lr = 1e-3, base_net_lr = 1e-3, extra_layers_lr = 1e-3, num_epochs = 100, momentum = 0.9, weight_decay = 5e-4, 
            debug_steps = 100, validation_epochs = 5,
            freeze_base_net = False, freeze_net = False,
            resume = None, base_net = None, pretrained_ssd = None,
            checkpoint_folder = "models/"):

        if freeze_base_net:
            logger.info("Freeze base net.")

            freeze_net_layers(self.net.base_net)
            params = itertools.chain(self.net.source_layer_add_ons.parameters(), self.net.extras.parameters(),
                                    self.net.regression_headers.parameters(), self.net.classification_headers.parameters())
            params = [
                {'params': itertools.chain(
                    self.net.source_layer_add_ons.parameters(),
                    self.net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    self.net.regression_headers.parameters(),
                    self.net.classification_headers.parameters()
                )}
            ]
        elif freeze_net:
            freeze_net_layers(self.net.base_net)
            freeze_net_layers(self.net.source_layer_add_ons)
            freeze_net_layers(self.net.extras)
            params = itertools.chain(self.net.regression_headers.parameters(), self.net.classification_headers.parameters())
            logger.info("Freeze all the layers except prediction heads.")
        else:
            params = [
                {'params': self.net.base_net.parameters(), 'lr': base_net_lr},
                {'params': itertools.chain(
                    self.net.source_layer_add_ons.parameters(),
                    self.net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    self.net.regression_headers.parameters(),
                    self.net.classification_headers.parameters()
                )}
            ]

        if resume:
            logger.info(f"Resume from the model {resume}")
            self.net.load(resume)
        elif base_net:
            logger.info(f"Init from base net {base_net}")
            self.net.init_from_base_net(base_net)
        elif pretrained_ssd:
            logger.info(f"Init from pretrained ssd {pretrained_ssd}")
            self.net.init_from_pretrained_ssd(pretrained_ssd)

        self.net.to(device)

        if criterion is None:
            criterion = MultiboxLoss(mobilenet_ssd_config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                    center_variance=0.1, size_variance=0.2, device=device)
        if optimizer is None:
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                        weight_decay=weight_decay)
        if scheduler is None:
            scheduler = CosineAnnealingLR(optimizer, 120)

        logger.info("Start training using CosineAnnealingLR scheduler.")

        for epoch in range(0, num_epochs):
            self.step(train_loader, criterion, optimizer, epoch=epoch,
                device=device, debug_steps=debug_steps)

            scheduler.step()

            if (epoch % validation_epochs == 0) or (epoch == num_epochs - 1):
                val_loss, val_regression_loss, val_classification_loss = self.eval(val_loader, criterion, device)
                logger.info(
                    f"Epoch: {epoch}, " +
                    f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Regression Loss {val_regression_loss:.4f}, " +
                    f"Validation Classification Loss: {val_classification_loss:.4f}"
                )
                model_path = os.path.join(checkpoint_folder, f"mobilenet-v2-ssd-lite-Epoch-{epoch}-Loss-{val_loss}.pth")
                self.net.save(model_path)

                logger.info(f"Saved model {model_path}")

    def eval(self, loader, criterion, device):
        self.net.eval()
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        num = 0
        for _, data in enumerate(loader):
            images, boxes, labels = data
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            num += 1

            with torch.no_grad():
                confidence, locations = self.net(images)
                regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
                loss = regression_loss + classification_loss

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
        return running_loss / num, running_regression_loss / num, running_classification_loss / num
