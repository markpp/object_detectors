import os
import numpy as np
import cv2
import torch
import random


# remove when implemented in sample_dataset.py
def label_unique(mask):
    new_mask = np.zeros(mask.shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i, cont in enumerate(contours):
        cv2.drawContours(new_mask, [cont], contourIdx=-1, color=(i+1), thickness=-1)
    return new_mask

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms

        img_dir = os.path.join(self.root,'PNGImages')
        self.img_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
        mask_dir = os.path.join(self.root,'PedMasks')
        self.mask_list = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

        if len(self.img_list) != len(self.mask_list):
            print("mismatch! number of rgb images: {} number of masks {}".format(len(img_list),len(mask_list)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image_id = os.path.basename(img_path).split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_path = self.mask_list[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            sample = self.transforms(image=img, mask=mask)
            img = sample["image"]
            mask = sample["mask"]

        target = {}

        # if all objects in mask have been given the same value
        #mask = label_unique(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        boxes = []
        num_objs = len(obj_ids)
        # if no objects exists
        if num_objs == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            for i, obj in enumerate(obj_ids):
                pos = np.where(masks[i])
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = np.array(boxes)

        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)

        target['labels'] = torch.ones((num_objs,), dtype=torch.int64)
        target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), dtype=torch.float32)

        img = img.transpose((2, 0, 1))
        img = img/255.0
        img[0] = (img[0] - 0.485)/0.229
        img[1] = (img[1] - 0.456)/0.224
        img[2] = (img[2] - 0.406)/0.225
        img = torch.from_numpy(img).float()
        return img, target
