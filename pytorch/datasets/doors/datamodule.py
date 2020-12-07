import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os, sys
from glob import glob
import cv2

sys.path.append('../../')
from datasets.doors.dataset import DoorDataset

import albumentations as Augment
from albumentations.pytorch.transforms import ToTensor

def train_transforms(img_height, img_width):
    return Augment.Compose([Augment.RandomScale(scale_limit=0.1, interpolation=cv2.INTER_NEAREST, always_apply=False, p=0.5),
                            Augment.PadIfNeeded(img_height, img_width, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
                            #Augment.RandomCropNearBBox(max_part_shift=0.3, always_apply=True),
                            Augment.RandomCrop(img_height, img_width, always_apply=True),
                            Augment.Flip(p=0.75),
                            Augment.Transpose(p=0.5),
                            ToTensor()],
                            bbox_params=Augment.BboxParams(format="albumentations", min_area=1,
                                                           min_visibility=0.5, label_fields=["labels"]))

#Augment.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
def test_transforms(img_height, img_width):
    return Augment.Compose([Augment.PadIfNeeded(img_height, img_width, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=255),
                            #Augment.RandomCropNearBBox(max_part_shift=0.3, always_apply=True),
                            Augment.RandomCrop(img_height, img_width, always_apply=True),
                            ToTensor()],
                            bbox_params=Augment.BboxParams(format="albumentations", min_area=1,
                                                           min_visibility=0.5, label_fields=["labels"]))
def collate_fn(batch):
    return list(zip(*batch))

class DoorDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size, num_workers=12, num_classes=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.num_classes = num_classes

        self.setup()

    #def prepare_data():
        #download, unzip here. anything that should not be done distributed
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = DoorDataset(self.data_dir,'train_list_reduced_cnn.txt',
                                          transforms=train_transforms(self.image_size,self.image_size),
                                          crop_size=self.image_size)
            self.data_val = DoorDataset(self.data_dir,'valid_list_reduced_cnn.txt',
                                        transforms=test_transforms(self.image_size,self.image_size),
                                        crop_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)


if __name__ == '__main__':
    image_size = 800
    dm = DoorDataModule(data_dir='/home/markpp/github/MapGeneralization/data/Public',
                        batch_size=16,
                        image_size=image_size)


    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.train_dataloader()):
        imgs, targets = batch
        for img, target in zip(imgs,targets):
            #img = img.mul(255).byte().numpy()
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # draw annotations
            #print("b {}".format(target["boxes"]))
            for b in target['boxes']:
                x1, y1, x2, y2 = b.tolist()
                #print([y1, x1, y2, x2])
                #map(int, bbox)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img,(x1, y1),(x2, y2),(0,255,0),2)

            output_dir = os.path.join(output_root,str(batch_id).zfill(6))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = "{}_id-{}.png".format(target['image_id'],str(sample_idx).zfill(6))
            cv2.imwrite(os.path.join(output_dir,filename),img)
            sample_idx = sample_idx + 1
        if batch_id > 0:
            break
