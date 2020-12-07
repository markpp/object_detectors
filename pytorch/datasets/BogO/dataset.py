import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import random
import albumentations as Augment
from skimage.exposure import match_histograms

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
        return boxes, labels

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

class BogODataset(Dataset):
    def __init__(self, root, transforms=None):
        super().__init__()

        self.root = root

        self.transforms = transforms

        self.defect_dir = os.path.join(root, 'defect')
        self.defect_list = sorted([os.path.join(self.defect_dir, f) for f in os.listdir(self.defect_dir) if f.endswith('.jpg')])

        self.ok_dir = os.path.join(root, 'ok')
        self.ok_list = sorted([os.path.join(self.ok_dir, f) for f in os.listdir(self.ok_dir) if f.endswith('.jpg')])

        self.img_list = self.ok_list + self.defect_list

        self.COPYPASTE = True

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        img_path = self.img_list[index]

        image_id = os.path.basename(img_path).split('.')[0]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)[:,:,0]
        cv2.imshow("before",image)

        if index < len(self.ok_list) and self.COPYPASTE:
            # copy-paste random defect
            defect_index = random.randint(0,len(self.defect_list)-1)
            defect_path = self.defect_list[defect_index]
            defect = cv2.imread(defect_path, cv2.IMREAD_COLOR)[:,:,0]
            #cv2.imshow("defect",defect)

            mask_path = defect_path.replace('jpg','png')

            defect_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # rotate, shear, scale
            #cv2.imshow("mask",defect_mask)

            # copy-past defect using mask
            roi = np.where(defect_mask > 0)
            # normalize color / crightness
            #defect[roi]
            #matched = match_histograms(defect[roi], image[roi], multichannel=False)

            cv2.imshow("org defect",defect)

            matched_defect = match_histograms(defect, image, multichannel=False).astype(np.uint8)
            cv2.imshow("matched_defect",matched_defect)

            blend_ratio = 0.5


            #matched = cv2.addWeighted(matched, blend_ratio, image, 1 - blend_ratio, 0)
            #cv2.imshow("blend",matched)


            image[roi] = defect[roi]
            cv2.imshow("after org",image)

            image[roi] = matched_defect[roi]
            cv2.imshow("after match",image)

        else:
            mask_path = img_path.replace('jpg','png')
            defect_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # DETR takes in data in coco format, but in this case yolo
        unique_mask  = label_unique(defect_mask)
        boxes, labels = mask2gt(unique_mask)
        if len(labels) > 0:
            # Area of bb
            area = boxes[:, 2] * boxes[:, 3]

            # We have a labels column it is multi object supported.
            #labels = records[self.target].values
            #labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            area = 0.0

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

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

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.float32)
        target['image_id'] = torch.tensor([index])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)

        #return torch.Tensor(image/255.0), target, image_id
        return image, target, image_id


if __name__ == '__main__':
    """
    Crops around defects(positives) and ok(negatives).

    Command:
        python main.py
    """

    dataset = BogODataset(root="/home/markpp/datasets/bo/train_1024",
                               transforms=train_transforms(512, 512))

    for i in range(len(dataset))[:]:
        img, target, image_id = dataset[i]


        #img[0] = img[0]*0.229 + 0.485
        '''
        img[1] = img[1]*0.224 + 0.456
        img[2] = img[2]*0.225 + 0.406
        img = img.mul(255).permute(1, 2, 0).byte().numpy()
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        '''
        img = img.mul(255).byte().numpy()
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        #print("crop {} x {}".format(h, w))
        for bbox in target['boxes']:
            # x, y, w, h
            left, top = bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2
            right, bottom = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
            # x, y, w, h
            #left, top = bbox[0], bbox[1]
            #right, bottom = bbox[0] + bbox[2], bbox[1] + bbox[3]
            print((left*w, top*h))
            print((right*w, bottom*h))
            bgr = cv2.rectangle(bgr, (left*w, top*h), (right*w, bottom*h), (0,255,0), 1)

            if bbox[2] < 0 or bbox[3] < 0:
                print(image_id)
                print(bbox)


        cv2.imshow('crop',bgr)

        key = cv2.waitKey()
        if key == 27:
            break



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
