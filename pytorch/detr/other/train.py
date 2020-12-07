import dataset
from pprint import pprint
import config

from torchvision import transforms as T
import pandas as pd
from tqdm import tqdm
#import model
from src.model import detr_model
from src.engine import train_fn, eval_fn
import numpy as np
from src.detr_loss import HungarianMatcher, SetCriterion
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def collate_fn(batch):
    return tuple(zip(*batch))

def run():
    #https://github.com/lessw2020/training-detr
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    train_dataset = dataset.DetectionDataset(root="/home/markpp/datasets/bo/train_1024")

    valid_dataset = dataset.DetectionDataset(root="/home/markpp/datasets/bo/val_1024")

    train_dataloader = DataLoader(train_dataset, batch_size=config.hparams.batch_size,
    shuffle=False, collate_fn=collate_fn)

    valid_dataloader = DataLoader(valid_dataset, batch_size=config.hparams.batch_size, shuffle=False,
                                collate_fn=collate_fn)

    print("Data Loaders created")

    detector = detr_model(n_classes=config.hparams.n_classes, n_queries=config.hparams.n_queries,
    backbone=config.hparams.backbone, pretrained=config.hparams.pretrained)

    print('''Model Created with backbone = {}, pretrained = {},
        number of classes = {}, number of queries = {}'''.format(config.hparams.backbone, config.hparams.pretrained,
        config.hparams.n_classes, config.hparams.n_queries))

    matcher = HungarianMatcher()
    weight_dict = {"loss_ce" : 1, "loss_bbox" : 1, "loss_giou" : 1}
    losses = ['labels', 'boxes', 'cardinality']

    optimizer = optim.Adam(detector.parameters(), lr=config.hparams.learning_rate)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    detector.to(device)

    criterion = SetCriterion(config.hparams.n_classes - 1, matcher, weight_dict, eos_coef=config.hparams.null_class_coef, losses=losses)
    criterion.to(device)

    max_loss = 99999

    print("------- Training Started ----- ")

    for epoch in tqdm(range(config.hparams.n_epochs)):
        print("Epoch = {}".format(epoch))
        train_loss = train_fn(train_dataloader, detector, criterion, optimizer, device)
        validation_loss = eval_fn(valid_dataloader, detector, criterion, device)

        if validation_loss.avg < max_loss:
            max_loss = validation_loss.avg
            print("Validation Loss reduced than previous stage. Saving new model")
            torch.save(detector.state_dict(), config.hparams.model_save_path)
            print('-' * 25)
            print("Model Trained and Saved to Disk")

if __name__ == "__main__":
    run()
