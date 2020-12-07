import argparse
import os
import random
import torch
import pytorch_lightning as pl

#https://github.com/pytorch/pytorch/issues/42300

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fine", type=str,
                    default='fine', help="Flag")
    ap.add_argument("-s", "--save", type=bool,
                    default=True, help="Save model")
    args = vars(ap.parse_args())


    cfg = {
        'num_classes': 1,
        'lr_epoch': (30, 40),
        'max_epoch': 50,
        'image_size': 600,
        'name': 'Spray',
    }

    from lightning_model import LightningDetector
    model = LightningDetector(cfg)

    import os, sys
    sys.path.append('../')

    from datasets.sprays.datamodule import SprayDataModule
    dm = SprayDataModule(data_dir='/home/markpp/datasets/teejet/iphone_data/frames',
                         batch_size=6,
                         cfg=cfg)
    dm.setup()


    trainer = pl.Trainer(gpus=1, max_epochs=25, limit_train_batches=1.0, limit_val_batches=1.0, accumulate_grad_batches=4)#, profiler=True)

    trainer.fit(model, dm)

    if args['save']:
        # save model
        output_dir = 'trained_models/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        #torch.save(model.model, os.path.join(output_dir,'model.pt'))
        torch.save(model.model.state_dict(), os.path.join(output_dir,'model_dict.pth'))
        torch.save(model.model, os.path.join(output_dir,'model.pt'))
