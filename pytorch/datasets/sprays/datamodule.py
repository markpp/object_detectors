import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os, sys
from glob import glob
import cv2
import random
import numpy as np

import albumentations as Augment
from albumentations.pytorch.transforms import ToTensor
import torchvision.transforms as transforms


SPRAY_CLASSES = ['blue']

def vis_data(images, targets, input_size):
    # vis data
    img = images[0].mul(255).byte().permute(1, 2, 0).numpy()[:, :, ::-1]

    '''
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = ((img * std + mean)*255).astype(np.uint8)
    '''
    #cv2.imwrite('1.jpg', img)

    #img_ = cv2.imread('1.jpg')
    img_ = img.copy()
    for point in targets[0]:
        x, y = point[:-1]
        x *= input_size
        y *= input_size
        cv2.circle(img_,(int(x), int(y)), 5, (0,255,0), -1)
    '''
    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
    '''
    cv2.imshow('img', img_)
    #cv2.waitKey(0)

def vis_heatmap(targets):
    # vis heatmap
    HW = targets.shape[1]
    h = int(np.sqrt(HW))
    for c in range(1):
        class_heatmap = targets[0, :, c].reshape(h, h)
        class_heatmap = cv2.resize(class_heatmap, (512, 512))
        cv2.imshow("class_heatmap", class_heatmap)
        x_heatmap = targets[1, :, c].reshape(h, h)
        x_heatmap = cv2.resize(x_heatmap, (512, 512))
        cv2.imshow("x_heatmap", x_heatmap)
        y_heatmap = targets[2, :, c].reshape(h, h)
        y_heatmap = cv2.resize(y_heatmap, (512, 512))
        cv2.imshow("y_heatmap", y_heatmap)
        weight_heatmap = targets[3, :, c].reshape(h, h)
        weight_heatmap = cv2.resize(weight_heatmap, (512, 512))
        cv2.imshow("weight_heatmap", weight_heatmap)
        #cv2.waitKey(0)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def basic_transforms(img_height, img_width, image_pad=200):
    return Augment.Compose([Augment.Resize(img_height, img_width, interpolation=cv2.INTER_NEAREST, always_apply=True),
                            #Augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True)
                            ], keypoint_params=Augment.KeypointParams(format='xy',label_fields=['class_labels'],remove_invisible=True))#ToTensor()

def train_transforms(img_height, img_width, image_pad=0):
    return Augment.Compose([#Augment.ToGray(p=1.0),
                            Augment.Rotate(limit=360, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=True),
                            Augment.RandomScale(scale_limit=[0.8,1.2], interpolation=1, always_apply=True),
                            #Augment.Crop(x_min=0, y_min=0, x_max=int(1080*0.8), y_max=int(1080*0.8),always_apply=True),
                            #Augment.CenterCrop(int(1080*0.8), int(1080*0.8), always_apply=True),
                            #Augment.RandomCrop(int(1080*0.7), int(1080*0.7), always_apply=True),
                            Augment.Resize(img_height, img_width, interpolation=cv2.INTER_NEAREST, always_apply=True),
                            Augment.RandomBrightnessContrast(p=1.0),
                            Augment.HorizontalFlip(p=0.5),
                            #Augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True)
                            ], keypoint_params=Augment.KeypointParams(format='xy',label_fields=['class_labels'],remove_invisible=True))#ToTensor()


class SprayDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size):
        super().__init__()
        self.batch_size = batch_size

        with open(os.path.join(data_dir,'det_train_json.txt'), 'r')  as f:
            self.train_list = f.read().splitlines()

        with open(os.path.join(data_dir,'det_val_json.txt'), 'r')  as f:
            self.test_list = f.read().splitlines()

        self.image_size = image_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            import os, sys
            sys.path.append('../../')
            from datasets.sprays.dataset import SprayDataset
            '''
            dataset = SprayDataset(root=self.dataset_root, transform=SSDAugmentation(self.cfg['min_dim'], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))
            split = int(len(dataset)*0.8)
            indices = torch.randperm(len(dataset)).tolist()
            self.data_train = torch.utils.data.Subset(dataset, indices[:split])
            self.data_val = torch.utils.data.Subset(dataset, indices[split:])

            '''
            self.data_train = SprayDataset(self.train_list,
                                           transforms=train_transforms(img_height=self.image_size,
                                                                       img_width=self.image_size)
                                           )#noise_transform=extra_transforms())
            self.data_val = SprayDataset(self.test_list,
                                         transforms=basic_transforms(img_height=self.image_size,
                                                                     img_width=self.image_size)
                                        )

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=detection_collate,
                          #pin_memory=True,
                          num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=detection_collate,
                          #pin_memory=True,
                          num_workers=6)


if __name__ == '__main__':

    image_size = 512
    num_classes = 1

    dm = SprayDataModule(data_dir='/home/markpp/datasets/teejet/Aabybro/iphone/',
                         batch_size=8,
                         image_size=image_size)

    dm.setup()


    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    import tools
    for batch_id, (images, targets) in enumerate(dm.train_dataloader()):
        print("batch_id {}".format(batch_id))
        targets = [label.tolist() for label in targets]
        #for img, tar in images, targets:
        vis_data(images, targets, image_size)

        targets = tools.gt_creator(input_size=image_size, stride=4, num_classes=num_classes, label_lists=targets)
        vis_heatmap(targets)

        key = cv2.waitKey(0)
        if key == 27:
            break
        '''
        for img, label in zip(images, targets):
            print(img)
            print(label)
            break
        break
        '''
        '''
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
        '''
