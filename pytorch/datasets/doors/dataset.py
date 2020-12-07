import os
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms.transforms import RandomCrop, ToTensor, Normalize, Compose
import random
import math

import cv2

#from converter import make_hm_regr, pred2box


def order_p1_p2(x1, y1, x2, y2):
    if x1 > x2:
        return x2, y2, x1, y1
    else:
        return x1, y1, x2, y2

'''
(1) binarize images
(2) compute number of crops based on image size
'''
#datasets: 'standard','centernet'
class DoorDataset(torch.utils.data.Dataset):
    def __init__(self, root, list, crop_size=1024, dataset='standard', transforms=None):
        self.root = root
        self.img_list = []
        with open(os.path.join(root,list)) as f:
            self.list = f.read().splitlines()
        for path in self.list:
            self.img_list.append(os.path.join(root,path))

        self.crop_size = crop_size
        self.transforms = transforms

        self.dataset = dataset
        # N (number of crops for this sample given the image size)
        self.N = 0

    def load_sample(self, idx):
        filename = os.path.basename(self.img_list[idx]).split('.')[0]
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        (T, img) = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img_h, img_w = img.shape[:2]

        # compute N
        self.N = math.ceil((img_h * img_w) / (self.crop_size * self.crop_size))

        boxes_path = self.img_list[idx].replace('/images/','/bboxes/').replace('.png','_boxes_image_format.txt')
        with open(boxes_path,"r") as boxes_file:
            bb_lines = boxes_file.readlines()

        boxes = []
        labels = []
        for bb_line in bb_lines:
            x1, y1, x2, y2 = bb_line.split(' ')
            x1, y1, x2, y2 = order_p1_p2(float(x1), float(y1), float(x2), float(y2))
            #x1, y1, x2, y2 = int(float(x1)*img_w), int(float(y1)*img_h), int(float(x2)*img_w), int(float(y2)*img_h)
            #print([x1, y1, x2, y2])
            boxes.append([x1, y1, x2, y2])
            labels.append(1)

        return img, boxes, labels, filename

    def __getitem__(self, idx):

        if self.N < 1:
            self.image, self.boxes, self.labels, self.filename = self.load_sample(idx)
        else:
            self.N -= 1

        if self.transforms:
            sample = {
                "image": self.image,
                "bboxes": self.boxes,
                "labels": self.labels,
            }
            sample = self.transforms(**sample)
            image = torch.as_tensor(sample["image"], dtype=torch.float32)
            boxes = torch.as_tensor(sample["bboxes"], dtype=torch.float32)
            labels =  torch.as_tensor(sample["labels"], dtype=torch.int64)
            filename = self.filename


        if self.dataset == 'centernet':
            target = {}
            if len(boxes):
                target["x"] = [x for x in boxes[0]]*self.crop_size
                target["w"] = [x1+(x2-x1)//2 for x1,x2 in zip(boxes[0],boxes[2])]*self.crop_size
                target["y"] = [y for y in boxes[1]]*self.crop_size
                target["h"] = [y1+(y2-y1)//2 for y1,y2 in zip(boxes[1],boxes[3])]*self.crop_size

            hm, regr = make_hm_regr(target)
            boxes, _ = pred2box(hm, regr)
            #print(boxes)
            cv2.imshow("tes",hm)
            cv2.waitKey(30)
            return torch.unsqueeze(image, 0), target
        else:
            target = {}
            #target["boxes"] = boxes#*self.crop_size
            #w, h = image.shape[:2]
            target["boxes"] = torch.as_tensor([[box[0]*self.crop_size,box[1]*self.crop_size,
                                                box[2]*self.crop_size,box[3]*self.crop_size] for box in boxes], dtype=torch.float32)
            target["labels"] = labels
            target["image_id"] = filename
            if len(labels) < 1:
                target["boxes"] = torch.as_tensor([[0.0,1.0,2.0,3.0]], dtype=torch.float32)
                target["labels"] =  torch.as_tensor([0], dtype=torch.int64)
            return image, target

    def __len__(self):
        return len(self.img_list)


'''
if __name__ == '__main__':

    dataset = DoorDataset(root='/home/markpp/github/MapGeneralization/data/Public',
                          list='train_list_reduced_cnn.txt')

    output_dir = "output/train"
    image_size = 1024

    for idx in range(len(dataset)):
        img, boxes = dataset[idx]
        img_h, img_w = img.shape[:2]

        # (1) check that there is room to crop the image
        if image_size >= img_w:
            right_pad = image_size - img_w
        else:
            right_pad = 0
        if image_size >= img_h:
            bottom_pad = image_size - img_h
        else:
            bottom_pad = 0
        img = cv2.copyMakeBorder(img, top=0, bottom=bottom_pad, left=0, right=right_pad, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

        # (2) determine how manyn times the image should be cropped
        N = math.ceil((img_h * img_w) / (image_size * image_size)) * 2

        for n in range(N):
            if img_w - (image_size-1) > 0 and img_h - (image_size-1) > 0:
                # (3) select a random x_min and y_min values inside the image
                x_min = random.randrange(0, img_w - (image_size-1))
                y_min = random.randrange(0, img_h - (image_size-1))
                x_max, y_max = x_min+image_size, y_min+image_size
                crop = img[y_min:y_max, x_min:x_max].copy()

                bbs = []
                # (4) check if bounding boxes are indside the crop
                for bb in boxes:
                    #if bb intersects the crop rect
                    if bb[0] > x_min and bb[1] > y_min and bb[2] < x_max and bb[3] < y_max:
                        bb_x_min, bb_y_min = bb[0]-x_min, bb[1]-y_min
                        bb_x_max, bb_y_max = bb[2]-x_min, bb[3]-y_min
                        #crop = cv2.rectangle(crop,(bb_x_min, bb_y_min),(bb_x_max, bb_y_max),(0,0,255),2)
                        bbs.append([bb_x_min/image_size,
                                    bb_y_min/image_size,
                                    bb_x_max/image_size,
                                    bb_y_max/image_size])

                if 1:#len(bbs) > 0:
                    filename = "idx-{}_n-{}_x_min-{}_y_min-{}.jpg".format(idx, n, x_min, y_min)
                    cv2.imwrite(os.path.join(output_dir,filename),crop)
                    with open(os.path.join(output_dir,filename.replace('jpg','txt')), 'w') as pred_file:
                        for bb in bbs:
                            pred_file.write('0 {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(bb[0],bb[1],bb[2],bb[3]))
'''
