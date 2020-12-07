from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import torch.nn as nn




def run_cli():

    pl.seed_everything(42)


    import os, sys
    sys.path.append('../')


    from datasets.VOC.vocdetection_datamodule import VOCDetectionDataModule
    datamodule = VOCDetectionDataModule(data_dir="/home/markpp/datasets/VOC",batch_size=12)
    '''

    from datasets.doors.datamodule import DoorDataModule
    datamodule = DoorDataModule(data_dir='/home/markpp/github/MapGeneralization/data/Public',
                                batch_size=4,
                                image_size=800)
    '''
    from detr_model import Detr
    model = Detr(pretrained=True, num_classes=datamodule.num_classes)

    #from torchsummary import summary
    #summary(model.model, torch.tensor((3, 800, 800)).cuda())

    trainer = pl.Trainer(gpus=1, limit_val_batches=0.2, max_epochs=100, accumulate_grad_batches=2, weights_summary='full')
    trainer.fit(model, datamodule)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.model.state_dict(), os.path.join(output_dir,"model.pt"))
    torch.save(model.model, os.path.join(output_dir,'final_model.pkl'))


if __name__ == "__main__":
    run_cli()
