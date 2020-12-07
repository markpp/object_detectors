import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import os
import cv2
import random
import pytorch_lightning as pl
import numpy as np

from datasets.BoODatasets import class_folders_dataset

import albumentations as Augment
from albumentations.pytorch.transforms import ToTensorV2

def train_transforms(img_height, img_width):
    return Augment.Compose([Augment.Rotate(limit=90, interpolation=1, border_mode=4, p=0.5),
                            Augment.VerticalFlip(p=0.5),
                            Augment.HorizontalFlip(p=0.5),
                            #Augment.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                            #                         rotate_limit=45, interpolation=1,
                            #                         border_mode=4, value=None, mask_value=None,
                            #                         always_apply=False, p=0.5),
                            Augment.RandomCrop(img_height, img_width, always_apply=True, p=1.0),
                            #Augment.RandomResizedCrop(height=img_height, width=img_width,
                            #                          scale=(0.9, 1.1), ratio=(1.0, 1.0),
                            #                          interpolation=1, always_apply=True, p=1.0),
                            #Augment.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5,
                            #                           val_shift_limit=5, p=0.5),
                            #Augment.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
                            #Augment.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                            Augment.Normalize(mean=(0.485, 0.456, 0.406),
                                               std=(0.229, 0.224, 0.225), always_apply=False),
                            ToTensorV2(p=1.0)],
                            p=1.0,
                            bbox_params=Augment.BboxParams(format="yolo",
                                                           min_area=1,
                                                           min_visibility=0,
                                                           label_fields=["labels"]))

def test_transforms(img_height, img_width):
    return Augment.Compose([Augment.RandomCrop(img_height, img_width, always_apply=True, p=1.0),
                            Augment.Normalize(mean=(0.485, 0.456, 0.406),
                                              std=(0.229, 0.224, 0.225), always_apply=False),
                            ToTensorV2(p=1.0)],
                            p=1.0,
                            bbox_params=Augment.BboxParams(format="yolo",
                                                           min_area=1,
                                                           min_visibility=0,
                                                           label_fields=["labels"]))

def collate_fn(batch):
    return tuple(zip(*batch))


class BoODataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, batch_size):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = class_folders_dataset(self.train_dir, transforms=train_transforms(512, 512))
            self.data_test_val = class_folders_dataset(self.test_dir, transforms=test_transforms(512, 512))
            n_sample = len(self.data_test_val)
            end_val_idx = int(n_sample * 0.5)
            self.data_val = Subset(self.data_test_val, range(0, end_val_idx))
            self.data_test = Subset(self.data_test_val, range(end_val_idx + 1, n_sample))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

def train(dm):

    for batch in dm.train_dataloader():
        print(batch[0])
        break

def test(dm):


    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.test_dataloader()):
        imgs, targets, image_ids = batch
        for i, sample in enumerate(zip(imgs, targets, image_ids)):
            img, target, image_id = sample

            img[0] = img[0]*0.229 + 0.485
            img[1] = img[1]*0.224 + 0.456
            img[2] = img[2]*0.225 + 0.406
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            h, w = img.shape[:2]
            #print("crop {} x {}".format(h, w))
            for bbox in target['boxes']:
                # x, y, w, h
                left, top = bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2
                right, bottom = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
                # x, y, w, h
                #left, top = bbox[0], bbox[1]
                #right, bottom = bbox[0] + bbox[2], bbox[1] + bbox[3]
                bgr = cv2.rectangle(bgr, (left*w, top*h), (right*w, bottom*h), (0,255,0), 1)

                if bbox[2] < 0 or bbox[3] < 0:
                    print(image_id)
                    print(bbox)

            filename = "batch_id-{}_sample_id-{}.jpg".format(batch_id,i)
            cv2.imwrite(os.path.join(output_dir,filename),bgr)

        '''
        imgs, labels, coords, names = batch
        for sample in zip(imgs, labels, coords, names):
            img, label, coord, name = sample

            img[0] = img[0] * 0.229 - 0.485
            img[1] = img[1] * 0.224 - 0.456
            img[2] = img[2] * 0.225 - 0.406
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            output_dir = os.path.join(output_root,name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename = "id-{}_x-{}_y-{}_l-{}.jpg".format(str(sample_idx).zfill(6),coord[0],coord[1],label)
            cv2.imwrite(os.path.join(output_dir,filename),img)
            sample_idx = sample_idx + 1
        '''

if __name__ == '__main__':
    datamodule = BoODataModule(train_dir='/home/markpp/datasets/bo/train_1024',
                               test_dir='/home/markpp/datasets/bo/val_1024',
                               batch_size=16)
    datamodule.setup()

    #train(datamodule)
    test(datamodule)
