import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import random
import albumentations as Augment


# remove when implemented in sample_dataset.py
def label_unique(mask):
    new_mask = np.zeros(mask.shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i, cont in enumerate(contours):
        cv2.drawContours(new_mask, [cont], contourIdx=-1, color=(i+1), thickness=-1)
    return new_mask

class BogODataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.root = root

        self.transform = transform

        self.defect_dir = os.path.join(root, 'defect')
        self.defect_list = sorted([os.path.join(self.defect_dir, f) for f in os.listdir(self.defect_dir) if f.endswith('.jpg')])

        self.ok_dir = os.path.join(root, 'ok')
        self.ok_list = sorted([os.path.join(self.ok_dir, f) for f in os.listdir(self.ok_dir) if f.endswith('.jpg')])

        self.img_list =  self.defect_list + self.ok_list


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img_path = self.img_list[idx]
        image_id = os.path.basename(img_path).split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)#[:,:,0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = img_path.replace('jpg','png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            sample = self.transform(image=img, mask=mask)
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
                area = (xmax - xmin) * (ymax - ymin)
                if xmin > xmax or ymin > ymax or area < 10.0:
                    print([xmin, ymin, xmax, ymax])
                    print(img_path)
                else:
                    boxes.append([xmin, ymin, xmax, ymax])
            if len(boxes) > 0:
                boxes = np.array(boxes)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)

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

        '''
        boxes, labels = mask2bboxes(unique_mask)

        target = {}

        if len(boxes) < 1:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            target['boxes'] = boxes
            target['labels'] = torch.zeros(1, dtype=torch.int64)
        else:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        target["masks"] = torch.as_tensor(defect_mask, dtype=torch.uint8)
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), dtype=torch.float32)

        image = image.transpose((2, 0, 1))
        image = image/255.0
        image[0] = (image[0] - 0.485)/0.229
        image[1] = (image[1] - 0.456)/0.224
        image[2] = (image[2] - 0.406)/0.225
        image = torch.from_numpy(image)
        image = image.float()
        return image, target
        '''
