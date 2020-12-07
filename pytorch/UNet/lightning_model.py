from argparse import ArgumentParser
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from unet import UNet


class SemSegment(pl.LightningModule):
    def __init__(
            self,
            datamodule: pl.LightningDataModule = None,
            lr: float = 0.01,
            num_classes: int = 19,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
    ):
        """
        Basic model for semantic segmentation. Uses UNet architecture by default.

        The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
        you will first need to download the KITTI dataset yourself. You can download the dataset `here.
        <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_

        Implemented by:

            - `Annika Brundyn <https://github.com/annikabrundyn>`_

        Args:
            datamodule: LightningDataModule
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            lr: learning (default 0.01)
        """
        super().__init__()

        assert datamodule
        self.datamodule = datamodule

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr

        self.model = UNet(num_classes=num_classes,
                          num_layers=self.num_layers,
                          features_start=self.features_start,
                          bilinear=self.bilinear)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = F.cross_entropy(out, mask, ignore_index=250)
        #self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = F.cross_entropy(out, mask, ignore_index=250)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]
