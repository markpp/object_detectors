import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os, sys
from glob import glob
import cv2
import numpy as np

sys.path.append('../')
from datasets.PennFudanPed.dataset import Dataset

import albumentations as Augment

# RCNN
#def collate_fn(batch):
#    return list(zip(*batch))

# SSD
def collate_fn(batch):
    images, targets = list(zip(*batch))
    return images, targets

def train_transforms(img_height, img_width):
    return Augment.Compose(
        [#Augment.Rotate(limit=15, interpolation=1, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
         Augment.Resize(img_height, img_width, always_apply=True),
         Augment.VerticalFlip(p=0.5),
         Augment.HorizontalFlip(p=0.5),
         #Augment.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
         #                         rotate_limit=45, interpolation=1,
         #                         border_mode=4, value=None, mask_value=None,
         #                         always_apply=False, p=0.5),
         #Augment.RandomCrop(img_height, img_width, always_apply=True, p=1.0),
         #Augment.RandomResizedCrop(height=img_height, width=img_width,
         #                          scale=(0.9, 1.1), ratio=(1.0, 1.0),
         #                          interpolation=1, always_apply=True, p=1.0),
         #Augment.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5,
         #                           val_shift_limit=5, p=0.5),
         #Augment.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
         #Augment.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
         #Augment.Normalize(mean=(0.485, 0.456, 0.406),
         #                  std=(0.229, 0.224, 0.225), always_apply=True),
         #Augment.Normalize(mean=(0.485), std=(0.229), always_apply=True)
         #ToTensor()
         ]
    )

def test_transforms(img_height, img_width):
    return Augment.Compose([Augment.RandomCrop(img_height, img_width, always_apply=True, p=1.0)])

class PennFudanPedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size


    #def prepare_data():
        #download, unzip here. anything that should not be done distributed
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data = Dataset(root=self.data_dir,
                                transforms=train_transforms(self.image_size, self.image_size))

            n_sample = len(self.data)
            end_train_idx = int(n_sample * 0.8)
            self.data_train = Subset(self.data, range(0, end_train_idx))
            self.data_val = Subset(self.data, range(end_train_idx + 1, n_sample))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=8, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn)

    @property
    def num_classes(self):
        """
        Return:
            2
        """
        return 2

if __name__ == '__main__':

    dm = PennFudanPedDataModule(data_dir='/home/markpp/datasets/PennFudanPed/',
                                batch_size=16,
                                image_size=800)
    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.train_dataloader()):
        imgs, targets = batch
        for i, (img, tar) in enumerate(zip(imgs,targets)):
            img[0] = img[0] * 0.229 + 0.485
            img[1] = img[1] * 0.224 + 0.456
            img[2] = img[2] * 0.225 + 0.406

            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            masks = tar['masks'].mul(255).byte().numpy()

            mask = np.zeros(masks.shape[1:], dtype=np.uint8)
            for m in masks:
                indices = np.where(m>0)
                mask[indices] = m[indices]

            output_dir = os.path.join(output_root,str(batch_id).zfill(6))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            bbs = tar['boxes'].numpy()
            for bb in bbs:
                if bb[0] >= bb[2] or bb[1] >= bb[3]:
                    print(bb)
                cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255,0,0), 2)

            filename = "id-{}.jpg".format(str(sample_idx).zfill(6))
            cv2.imwrite(os.path.join(output_dir,filename),img)
            filename = "id-{}.png".format(str(sample_idx).zfill(6))
            cv2.imwrite(os.path.join(output_dir,filename),mask)
            sample_idx = sample_idx + 1

        break
