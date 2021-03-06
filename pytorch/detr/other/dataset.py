import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import random
import albumentations as Augment

def mask2gt(mask):
    # instances are encoded as different colors
    obj_ids = np.unique(mask)

    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]

    boxes, labels = [], []
    # if no objects exists
    if len(obj_ids) == 0:
        return boxes

    for i, obj in enumerate(obj_ids):
        pos = np.where(masks[i])
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        ymin, ymax = np.min(pos[0]), np.max(pos[0])

        # c_x, c_y, w, h
        w, h = xmax-xmin, ymax-ymin
        boxes.append([float(xmin + w/2.0)/mask.shape[1], float(ymin + h/2.0)/mask.shape[0],
                      float(w)/mask.shape[1], float(h)/mask.shape[0]])

        #boxes.append([float(xmin)/mask.shape[1], float(ymin)/mask.shape[0],
        #              float(xmax)/mask.shape[1], float(ymax)/mask.shape[0]]))
        #boxes.append([float(xmin), float(ymin),
        #              float(xmax)/mask.shape[1], float(ymax)/mask.shape[0]])
        #              float(xmax-xmin), float(ymax-ymin)])
        labels.append(0)

    return np.array(boxes), labels

def label_unique(mask):
    new_mask = np.zeros(mask.shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i, cont in enumerate(contours):
        cv2.drawContours(new_mask, [cont], contourIdx=-1, color=(i+1), thickness=-1)
    return new_mask

class DetectionDataset(Dataset):
    def __init__(self, root, transforms=None):
        super().__init__()

        self.root = root

        self.transforms = transforms

        self.defect_dir = os.path.join(root, 'defect')
        self.defect_list = sorted([os.path.join(self.defect_dir, f) for f in os.listdir(self.defect_dir) if f.endswith('.jpg')])

        self.ok_dir = os.path.join(root, 'ok')
        self.ok_list = sorted([os.path.join(self.ok_dir, f) for f in os.listdir(self.ok_dir) if f.endswith('.jpg')])

        self.img_list = self.ok_list + self.defect_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        img_path = self.img_list[index]

        image_id = os.path.basename(img_path).split('.')[0]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #cv2.imshow("before",image)

        if index < len(self.ok_list):
            # copy-paste random defect
            defect_index = random.randint(0,len(self.defect_list)-1)
            defect_path = self.defect_list[defect_index]
            defect = cv2.imread(defect_path, cv2.IMREAD_COLOR)
            #cv2.imshow("defect",defect)

            mask_path = defect_path.replace('jpg','png')

            defect_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # rotate, shear, scale
            #cv2.imshow("mask",defect_mask)

            # copy-past defect using mask
            roi = np.where(defect_mask > 0)
            # normalize color / crightness
            #defect[roi]
            image[roi] = defect[roi]
        else:
            mask_path = img_path.replace('jpg','png')
            defect_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        #cv2.imshow("after",image)
        #cv2.waitKey()

        # DETR takes in data in coco format, but in this case yolo
        unique_mask  = label_unique(defect_mask)

        boxes, labels = mask2gt(unique_mask)

        # Area of bb
        area = boxes[:, 2] * boxes[:, 3]
        area = torch.as_tensor(area, dtype=torch.float32)

        # We have a labels column it is multi object supported.
        #labels = records[self.target].values
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)


        if self.transforms:
            sample = {
                "image": image,
                "bboxes": boxes,
                "labels": labels,
            }
            sample = self.transforms(**sample)
            image = sample["image"]
            boxes = sample["bboxes"]
            labels = sample["labels"]

            #print(boxes.shape)
            #print(boxes.dtype)

            # do the bounding boxes

            #_, h, w = image.shape
            #boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'], rows=h, cols=w)

        # Normalize the bounding boxes
        #_, h, w = image.shape
        #boxes = Augment.augmentations.bbox_utils.normalize_bboxes(
        #    sample["boxes"], rows=h, cols=w
        #)

        #image = image.transpose((2, 0, 1))
        # Scale down the pixel values of image
        #image /= 255.0
        #image[0] = (image[0] - 0.485)/0.229
        #image[1] = (image[1] - 0.456)/0.224
        #image[2] = (image[2] - 0.406)/0.225
        #image = torch.from_numpy(image)



        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area

        return image, target, image_id


from albumentations.pytorch.transforms import ToTensorV2

def train_transforms(img_height, img_width):
    return Augment.Compose(
        [Augment.Rotate(limit=90, interpolation=1, border_mode=4, p=0.5),
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
         bbox_params=Augment.BboxParams(format="yolo", min_area=1,
                                        min_visibility=0, label_fields=["labels"]),
    )


if __name__ == '__main__':
    """
    Crops around defects(positives) and ok(negatives).

    Command:
        python main.py
    """




    dataset = DetectionDataset(root="/home/markpp/datasets/bo/val_1024",
                               transforms=train_transforms(512, 512))

    for i in range(len(dataset))[:]:
        img, target, image_id = dataset[i]

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


        cv2.imshow('crop',bgr)

        key = cv2.waitKey()
        if key == 27:
            break
