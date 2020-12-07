import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


import os, sys
from glob import glob
import cv2
from PIL import Image

sys.path.append('../')
from BogO.segmentation_dataset import BogODataset

import albumentations as Augment

from albumentations.pytorch.transforms import ToTensorV2, ToTensor

def collate_fn(batch):
    return list(zip(*batch))

def train_transforms(img_height, img_width):
    return Augment.Compose(
        [#Augment.Rotate(limit=90, interpolation=1, border_mode=4, p=0.5),
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

class BogODataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size


    #def prepare_data():
        #download, unzip here. anything that should not be done distributed
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = BogODataset(root=os.path.join(self.data_dir,'train_1024'),
                                          transform=train_transforms(self.image_size, self.image_size))
            self.data_val = BogODataset(root=os.path.join(self.data_dir,'val_1024'))#,
                                        #transform=test_transforms(self.image_size, self.image_size))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn)


if __name__ == '__main__':

    dm = BogODataModule(data_dir='/home/markpp/datasets/bo/',
                        batch_size=16,
                        image_size=512)
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
        for i, img in enumerate(imgs):
            img[0] = img[0] * 0.229 + 0.485
            img[1] = img[1] * 0.224 + 0.456
            img[2] = img[2] * 0.225 + 0.406

            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            output_dir = os.path.join(output_root,str(batch_id).zfill(6))
            #if not os.path.exists(output_dir):
            #    os.makedirs(output_dir)


            bbs = targets[i]['boxes'].numpy()
            for bb in bbs:
                if bb[0] >= bb[2] or bb[1] >= bb[3]:
                    print(bb)
                #cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255,0,0), 2)

            '''
            filename = "id-{}.jpg".format(str(sample_idx).zfill(6))
            cv2.imwrite(os.path.join(output_dir,filename),img)
            filename = "id-{}.png".format(str(sample_idx).zfill(6))
            cv2.imwrite(os.path.join(output_dir,filename),targets[i]['masks'].numpy())
            sample_idx = sample_idx + 1
            '''
        #if batch_id > 2:
            #break
