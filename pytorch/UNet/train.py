from argparse import ArgumentParser
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from unet import UNet
from lightning_model import SemSegment

def cli_main():
    from kitti_datamodule import KittiDataModule

    pl.seed_everything(1234)

    #parser = ArgumentParser()

    # trainer args
    #parser = pl.Trainer.add_argparse_args(parser)

    # model args
    #parser = SemSegment.add_model_specific_args(parser)

    # data
    dm = KittiDataModule(data_dir="/home/markpp/datasets/kitti",batch_size=1)

    model = SemSegment(datamodule=dm,
                       lr = 0.01,
                       num_classes = 19,
                       num_layers = 5,
                       features_start = 64,
                       bilinear = False)

    # train
    trainer = pl.Trainer(gpus=1,max_epochs=50,accumulate_grad_batches=16)
    trainer.fit(model)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.model.state_dict(), os.path.join(output_dir,"model.pt"))
    torch.save(model.model, os.path.join(output_dir,'final_model.pkl'))

if __name__ == '__main__':
    cli_main()
