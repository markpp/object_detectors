import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
import pytorch_lightning as pl

import detr_loss


from data_loader import prepare_data_from_list, standard_dataloader

class LightningDetr(pl.LightningModule):

    def __init__(self, hparams, pretrained=True):
      super().__init__()

      self.hparams = hparams


      self.model = torch.hub.load('facebookresearch/detr', self.hparams.backbone, pretrained=pretrained)


      self.in_features =  self.model.class_embed.in_features

      self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.hparams.num_classes)
      self.model.num_queries = self.hparams.num_queries

      self.matcher = detr_loss.HungarianMatcher()
      self.weight_dict = {"loss_ce" : 1, "loss_bbox" : 1, "loss_giou" : 1}
      self.losses = ['labels', 'boxes', 'cardinality']
      self.criterion = detr_loss.SetCriterion(config.hparams.num_classes - 1,
                                              self.matcher, self.weight_dict,
                                              eos_coef=config.hparams.null_class_coef,
                                              losses=self.losses)


    def forward(self, x):
        return self.model(x)

    # create folds from training set
    def prepare_data(self):
        # split the dataset in train and test set
        self.data_train = prepare_data_from_list(os.path.join(self.hparams.data_path,self.hparams.train_list), crop_size=self.hparams.input_size)
        self.data_val = prepare_data_from_list(os.path.join(self.hparams.data_path,self.hparams.val_list), crop_size=self.hparams.input_size)

    def train_dataloader(self):
        return standard_dataloader(self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_workers)

    def val_dataloader(self):
        return standard_dataloader(self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)
        loss_dict = self.criterion(output, target)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        epoch_loss = torch.stack([x['batch_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': epoch_loss}
        return {'train_loss': epoch_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)
        loss_dict = self.criterion(output, target)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    '''
    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)
        print(output)
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}
    '''
