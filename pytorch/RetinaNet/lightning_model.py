import argparse
import logging
from typing import Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from retinanet import Retinanet
from utils import collate_fn, load_obj


class RetinaNetModel(pl.LightningModule):
    """
    Lightning Class to wrap the RetinaNet Model.
    So that it can be trainer with LightningTrainer.

    Args:
      haprams (`DictConfig`) : A `DictConfig` that stores the configs for training .
    """

    def __init__(self, hparams: Union[DictConfig, argparse.Namespace]):
        super(RetinaNetModel, self).__init__()
        self.hparams = hparams
        self.net = Retinanet(**hparams.model, logger=logging.getLogger("lightning"))

        #add learning_rate to hparams dictionary
        self.hparams.learning_rate = self.hparams.optimizer.params.lr
        self.save_hyperparameters(hparams)

    def forward(self, xb, *args, **kwargs):
        output = self.net(xb)
        return output

    def configure_optimizers(self, *args, **kwargs):
        opt = self.hparams.optimizer.class_name
        self.optimizer = load_obj(opt)(self.net.parameters(), **self.hparams.optimizer.params)
        if self.hparams.scheduler.class_name is None:
            return [self.optimizer]

        else:
            schedps = self.hparams.scheduler
            __scheduler = load_obj(schedps.class_name)(self.optimizer, **schedps.params)
            if not self.hparams.scheduler.monitor:
                self.scheduler = {"scheduler": __scheduler,"interval": schedps.interval,"frequency": schedps.frequency,}
            else:
                self.scheduler = {"scheduler": __scheduler,"interval": schedps.interval, "frequency": schedps.frequency,"monitor": schedps.monitor,}

            return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch  # unpack the one batch from the DataLoader
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets
        # Calculate Losses {regression_loss , classification_loss}
        loss_dict = self.net(images, targets)
        # Calculate Total Loss
        losses = sum(loss for loss in loss_dict.values())
        return {"loss": losses, "log": loss_dict, "progress_bar": loss_dict}

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch  # unpack the one batch from the DataLoader
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets
        # Calculate Losses {regression_loss , classification_loss}
        loss_dict = self.net(images, targets)
        # Calculate Total Loss
        loss = sum(loss for loss in loss_dict.values())
        loss = torch.as_tensor(loss)
        logs = {"val_loss": loss}
        return {"val_loss": loss,"log": logs,"progress_bar": logs,}

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.net.predict(images)
        res = {t["image_id"].item(): o for t, o in zip(targets, outputs)}
        self.test_evaluator.update(res)
        return {}

    def test_epoch_end(self, outputs, *args, **kwargs):
        self.test_evaluator.accumulate()
        self.test_evaluator.summarize()
        metric = self.test_evaluator.coco_eval["bbox"].stats[0]
        metric = torch.as_tensor(metric)
        logs = {"AP": metric}
        return {"AP": metric,"log": logs,"progress_bar": logs,}
