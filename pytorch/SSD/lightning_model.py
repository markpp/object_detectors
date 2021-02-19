import torch
import pytorch_lightning as pl

import torchvision

import numpy as np
import cv2

from SSD import Net, MultiBoxLoss


'''
    def training_step(self, batch, batch_idx):
        xs, ys = batch
        xs  = [ torch.cuda.FloatTensor(x) for x in xs ]
        images = torch.stack(xs, dim=0)
        targets  = [ torch.cuda.FloatTensor(y) for y in ys ]
        outputs = self(images)
        loss_l, loss_c = criterion(outputs, targets)
        loss = loss_l + loss_c
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        xs  = [ torch.cuda.FloatTensor(x) for x in xs ]
        images = torch.stack(xs, dim=0)
        targets  = [ torch.cuda.FloatTensor(y) for y in ys ]
        outputs = self(images)
        loss_l, loss_c = criterion(outputs, targets)
        loss = loss_l + loss_c
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        xs  = [ torch.cuda.FloatTensor(x) for x in xs ]
        images = torch.stack(xs, dim=0)
        targets  = [ torch.cuda.FloatTensor(y) for y in ys ]
        outputs = self(images)
        criterion = MultiBoxLoss(num_classes=self.num_classes, overlap_thresh=0.5, neg_pos=3)
        loss_l, loss_c = criterion(outputs, targets)
        loss = loss_l + loss_c
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3,momentum=0.9, weight_decay=5e-4)
        return optimizer

'''

class LightningModel(pl.LightningModule):

    def __init__(self,num_classes: int = 2):
        super().__init__()

        self.model = Net(num_classes=num_classes)
        self.criterion = MultiBoxLoss(num_classes=num_classes, overlap_thresh=0.5, neg_pos=3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        outputs = self.model(images)
        loss_l, loss_c = criterion(outputs, targets)
        loss = loss_l + loss_c
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        outputs = self.model(images)
        loss_l, loss_c = criterion(outputs, targets)
        loss = loss_l + loss_c

        #if batch_idx == 0:
            #self.logger.experiment.add_image("images", _plot_boxes(images, targets, outs), self.current_epoch, dataformats='HWC')
            #print(outs[0]['masks'].shape)
            #self.logger.experiment.add_image("masks", outs[0]['masks'], self.current_epoch, dataformats='CHW')

        #iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        #return {"val_iou": iou}

    #def validation_epoch_end(self, outs):
    #    avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
    #    self.log('val_iou', avg_iou, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=0.0001,
            momentum=0.9,
            weight_decay=0.005,
        )
