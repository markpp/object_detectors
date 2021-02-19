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

    '''
    Collection of self-contained dectectors and acompanying datamodules

    TODO: add paper links and sort by date
    '''

    '''
    # bounding box dectectors #
    (-) SSD:
    (DONE) FasterRCNN:
    (WIP) YOLOv5: fast, easy
    (DONE) detr: slow training, flexible, global interactions?
    (WIP) EffecientDet: scalable
    (WIP) CenterNet: flexible and configurable
    (WIP) RetinaNet (Focal Loss for Dense Object Detection): first to add a focal loss addressing the foreground-background class imbalance problem in one-stage detectors. The focal loss lowers the loss contributed by “easy” negative samples, instead focusing on “hard” samples.
    '''


    '''
    # semantic segmentation #
    (DONE) UNet
    (DONE) MaskRCNN
    (WIP) DeepLab:
    '''


    '''
    Run training scripts from here by (1) creating a datamodule, (2) a training
    config, and (3) calling the training function for each detector.
    '''

    dm =

    cfg = dm.classes

    from UNet.train import train
    train(cfg)
