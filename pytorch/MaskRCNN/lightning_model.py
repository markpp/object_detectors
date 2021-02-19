import torch
import pytorch_lightning as pl

import torchvision

import numpy as np
import cv2

from MaskRCNN import create_model, _evaluate_iou, _plot_boxes


class LightningModel(pl.LightningModule):

    def __init__(self,num_classes: int = 2):
        super().__init__()

        self.model = create_model(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)

        if batch_idx == 0:
            self.logger.experiment.add_image("images", _plot_boxes(images, targets, outs), self.current_epoch, dataformats='HWC')
            #print(outs[0]['masks'].shape)
            #self.logger.experiment.add_image("masks", outs[0]['masks'], self.current_epoch, dataformats='CHW')

        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        self.log('val_iou', avg_iou, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=0.0001,
            momentum=0.9,
            weight_decay=0.005,
        )
