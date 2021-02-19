import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

def remove_samples_without_annotations(dataset):
    ids = []
    for ds_idx, sample in enumerate(dataset):
        if sample[0] is not None:
            ids.append(ds_idx)
    print("removed {} empty".format(len(ids)))
    return torch.utils.data.Subset(dataset, ids)

class CustomDataset(torch.utils.data.Dataset):
    # set preload = False if insufficient memoty for entire dataset
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.img_list = sorted([f for f in os.listdir(root) if f.endswith('.jpg')])
        self.mask_list = sorted([f for f in os.listdir(root) if f.endswith('.png')])
        #self.img_list = sorted([f for f in os.listdir(os.path.join(root, "rgb")) if f.endswith('.jpg')])
        #self.mask_list = sorted([f for f in os.listdir(os.path.join(root, "masks")) if f.endswith('.png')])
        #self.img_list = list(sorted(os.listdir(os.path.join(root, "rgb"))))
        #self.mask_list = list(sorted(os.listdir(os.path.join(root, "masks"))))

        # check that list entries match
        if len(self.img_list) != len(self.mask_list):
            print("mismatch! number of rgb images: {} number of masks {}".format(len(img_list),len(mask_list)))


    def load_sample(self, idx):
        img_path = os.path.join(self.root, self.img_list[idx])
        mask_path = os.path.join(self.root, self.mask_list[idx])
        #img_path = os.path.join(self.root, "rgb", self.img_list[idx])
        #mask_path = os.path.join(self.root, "masks", self.mask_list[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img, mask = self.transforms(img, mask)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        # if no objects exists
        if num_objs == 0:
            print(self.img_list[idx])
            return None, None

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __getitem__(self, idx):
        image, target = self.load_sample(idx)

        #if self.transforms is not None:
            #image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.img_list)
