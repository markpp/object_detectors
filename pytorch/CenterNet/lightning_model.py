import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

import tools

class LightningDetector(pl.LightningModule):

    def __init__(self, hparams, pretrained=False):
      super().__init__()
      self.hparams = hparams

      # init model
      from centernet import CenterNet
      self.model = CenterNet(input_size=self.hparams.image_size, num_classes=self.hparams.num_classes, trainable=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer#, StepLR(optimizer, step_size=1)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        targets = [label.tolist() for label in targets]
        targets = tools.gt_creator(self.hparams.image_size, self.model.stride, self.hparams.num_classes, targets)
        targets = torch.tensor(targets, device=self.device).float()

        print(images.shape)
        cls_loss, txty_loss, total_loss = self.model(images, target=targets)

        return {'loss': total_loss, 'cls_loss': cls_loss, 'txty_loss': txty_loss}

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_cls_loss = torch.stack([x['cls_loss'] for x in outputs]).mean()
        train_txty_loss = torch.stack([x['txty_loss'] for x in outputs]).mean()
        self.log('train_loss', train_loss, on_epoch=True)
        self.log('train_cls_loss', train_cls_loss, on_epoch=True)
        self.log('train_txty_loss', train_txty_loss, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        targets = [label.tolist() for label in targets]
        targets = tools.gt_creator(self.hparams.image_size, self.model.stride, self.hparams.num_classes, targets)
        targets = torch.tensor(targets, device=self.device).float()

        cls_loss, txty_loss, total_loss = self.model(images, target=targets)

        return {'loss': total_loss, 'cls_loss': cls_loss, 'txty_loss': txty_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_cls_loss = torch.stack([x['cls_loss'] for x in outputs]).mean()
        val_txty_loss = torch.stack([x['txty_loss'] for x in outputs]).mean()
        self.log('val_loss', val_loss, on_epoch=True)
        self.log('val_cls_loss', val_cls_loss, on_epoch=True)
        self.log('val_txty_loss', val_txty_loss, on_epoch=True)
