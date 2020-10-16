import argparse
import os

import torch
import pytorch_lightning as pl

import config


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

    trainer = pl.Trainer(gpus=1, max_epochs=100)#, profiler=True)

    from lightning_model import LightningDetr
    model = LightningDetr(config.hparams)

    trainer.fit(model)

    if args['save']:
        # save model
        output_dir = 'trained_models/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        #torch.save(model.model, os.path.join(output_dir,'model.pt'))
        torch.save(model.model.state_dict(), os.path.join(output_dir,'model_dict.pth'))
