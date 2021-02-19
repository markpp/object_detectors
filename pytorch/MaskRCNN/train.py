import argparse
import os
import random
import torch
import pytorch_lightning as pl


if __name__ == '__main__':
    """
    Trains

    Command:
        python train.py
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
        'max_epoch': 200,
        'image_size': 512,
        'batch_size': 8,
        'name': 'Spray',
        'device': 'cuda',
    }

    pl.seed_everything(42)

    import os, sys
    sys.path.append('../')

    '''
    from datasets.PennFudanPed.datamodule import PennFudanPedDataModule
    datamodule = PennFudanPedDataModule(data_dir='/home/markpp/datasets/PennFudanPed/',
                                        batch_size=4,
                                        image_size=800)
    '''
    from datasets.BogO.segmentation_datamodule import BogODataModule

    datamodule = BogODataModule(data_dir='/home/markpp/datasets/bo/',
                        batch_size=4,
                        image_size=1024)

    num_classes = datamodule.num_classes

    from lightning_model import LightningModel
    model = LightningModel(num_classes)

    trainer = pl.Trainer(gpus=1, max_epochs=250, accumulate_grad_batches=4)#, weights_summary='full') #val_percent_check=0.1,
    trainer.fit(model, datamodule)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.model.state_dict(), os.path.join(output_dir,"model.pt"))
    torch.save(model.model, os.path.join(output_dir,'final_model.pkl'))
