import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os, sys
from glob import glob
import cv2
import random

import albumentations as Augment
from albumentations.pytorch.transforms import ToTensor

import torchvision.transforms as transforms

def basic_transforms(img_height, img_width, image_pad=0):
    return Augment.Compose([#Augment.ToGray(p=1.0),
                            #Augment.Resize(img_height+image_pad, img_width+image_pad, interpolation=cv2.INTER_NEAREST, always_apply=True),
                            #Augment.RandomCrop(img_height, img_width, always_apply=True),
                            Augment.HorizontalFlip(p=0.5),
                            Augment.RandomBrightnessContrast(p=1.0),
                            ])#ToTensor()

def extra_transforms():
    return Augment.Compose([Augment.GaussNoise(p=0.75),
                            Augment.CoarseDropout(p=0.5),])

from dataset import SprayDataset

class SprayDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size, image_pad=0):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size

        self.prepare_data(data_dir)

    def prepare_data(self, root_path):
        #download, unzip here. anything that should not be done distributed
        folders = list(filter(os.path.isdir, [os.path.join(root_path, f) for f in sorted(os.listdir(root_path))]))[:72] # 72 = last front

        random.shuffle(folders)
        train_test_split = int(len(folders)*0.8)

        train_lists = []
        for folder in folders[:train_test_split]:
            train_lists.append([os.path.join(folder,f) for f in sorted(os.listdir(folder)) if f.endswith('.json')])
        self.train_list = [json for json_list in train_lists for json in json_list]

        test_lists = []
        for folder in folders[train_test_split:]:
            test_lists.append([os.path.join(folder,f) for f in sorted(os.listdir(folder)) if f.endswith('.json')])
        self.test_list = [json for json_list in test_lists for json in json_list]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = SprayDataset(self.train_list,
                                           crop_size=self.image_size,
                                           transforms=basic_transforms(img_height=self.image_size,
                                                                       img_width=self.image_size),
                                           )#noise_transform=extra_transforms())
            self.data_val = SprayDataset(self.test_list,
                                         crop_size=self.image_size,)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=8)

    '''
    # create folds from training set
    def prepare_data(self):
        # split the dataset in train and test set
        self.data_train = prepare_data_from_list(os.path.join(self.hparams.data_path,self.hparams.train_list), crop_size=self.hparams.input_size)
        self.data_val = prepare_data_from_list(os.path.join(self.hparams.data_path,self.hparams.val_list), crop_size=self.hparams.input_size)

    def train_dataloader(self):
        return standard_dataloader(self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_workers)

    def val_dataloader(self):
        return standard_dataloader(self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_workers)
    '''
if __name__ == '__main__':

    dm = SprayDataModule(data_dir='/home/markpp/datasets/teejet/iphone_data/frames/21-10-20',
                         batch_size=16,
                         image_size=240)

    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.train_dataloader()):
        if len(batch) == 2:
            imgs, labels = batch
            for img, label in zip(imgs, labels):
                img = img.mul(255).permute(1, 2, 0).byte().numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(img, "batch_id {}, label {}".format(batch_id,label), (20,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                output_dir = os.path.join(output_root,str(batch_id).zfill(6))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = "id-{}.png".format(str(sample_idx).zfill(6))
                cv2.imwrite(os.path.join(output_dir,filename),img)
                sample_idx = sample_idx + 1
            if batch_id > 1:
                break
        else:
            imgs = batch
            for img in imgs:
                img = img.mul(255).permute(1, 2, 0).byte().numpy()
                output_dir = os.path.join(output_root,str(batch_id).zfill(6))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = "id-{}.png".format(str(sample_idx).zfill(6))
                cv2.imwrite(os.path.join(output_dir,filename),img)
                sample_idx = sample_idx + 1
            if batch_id > 1:
                break
