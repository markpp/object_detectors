import os
import numpy as np
import torch
import torch.utils.data

import cv2
import random
import math
import json


class SprayDataset(torch.utils.data.Dataset):
    def __init__(self, json_list, transforms=None):
        self.transform = transforms
        self.json_list = json_list

    def load_sample(self, label_path):
        with open(label_path, 'r') as f:
            json_labels = json.loads(f.read())

        if json_labels is not None:
            image_path = label_path.replace('.json','.jpg')
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w = img.shape[:2]
            #print("img_h {}, img_w {}".format(img_h, img_w))
            top_points, labels = [], []
            for json_label in json_labels:
                x, y = json_label['keypoints']['top']
                top_points.append([x*img_w, y*img_h])
                labels.append([0])
            return img, top_points, labels
        else:
            print('ov failed to load a label')

    def compute_crop_box(self, img_w, img_h, center):
        x, y = center
        cv2.imshow("output",img)


    def __getitem__(self, idx):
        img, target, class_labels = self.load_sample(self.json_list[idx])

        if self.transform:
            #crop_box = self.compute_crop_box(w, h, target[random.randint(0,len(target)-1)])
            sample = self.transform(image=img, keypoints=target, class_labels=class_labels)
            img = np.array(sample['image'])
            class_labels = sample['class_labels']
            target = sample['keypoints']

        if len(target):
            target = torch.cat((torch.as_tensor(target, dtype=torch.float32),torch.as_tensor(class_labels, dtype=torch.float32)), 1)
            # Normalize coords
            w, h = img.shape[:2]
            target[:,0] /= w
            target[:,1] /= h
        else:
            target = torch.as_tensor(target, dtype=torch.float32)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)/255.0

        target = torch.as_tensor(target)
        return img, target


    def __len__(self):
        return len(self.json_list)


if __name__ == '__main__':
    root_path = '/home/markpp/datasets/teejet/iphone_data/frames/21-10-20'

    folders = list(filter(os.path.isdir, [os.path.join(root_path, f) for f in sorted(os.listdir(root_path))]))

    train_test_split = int(len(folders)*0.8)

    train_lists = []
    for folder in folders[:train_test_split]:
        train_lists.append([os.path.join(folder,f) for f in sorted(os.listdir(folder)) if f.endswith('.json')])
    train_list = [json for json_list in train_lists for json in json_list]

    test_lists = []
    for folder in folders[train_test_split:]:
        test_lists.append([os.path.join(folder,f) for f in sorted(os.listdir(folder)) if f.endswith('.json')])
    test_list = [json for json_list in test_lists for json in json_list]

    dataset = SprayDataset(train_list)

    while(1):
        i = random.randint(0,len(dataset))-1
        img, label = dataset[i]
        #for i, data in enumerate(dataset):
            #img, label = data
        label = label.numpy()
        img = img.mul(255).permute(1, 2, 0).byte().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, "img {}, label {}".format(i,label), (20,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow("output",img)
        key = cv2.waitKey()
        if key == 27:
            break
'''

    def rot_and_crop(self, img, values):
        image_h, image_w, _ = img.shape
        cX, cY = image_w//2, image_h//2


        # original
        top_x, top_y = int(values[0]*self.image_w), int(values[1]*self.image_h)
        left_x, left_y = int(values[2]*self.image_w), int(values[3]*self.image_h)
        right_x, right_y = int(values[4]*self.image_w), int(values[5]*self.image_h)
        if self.show:
            img = cv2.circle(img, (top_x, top_y), 2, (0,255,255), 5)

        if self.enable_rot:
            angle = random.randint(0,180)
            M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

            # compute the new bounding dimensions of the image
            #cos = np.abs(M[0, 0])
            #sin = np.abs(M[0, 1])
            #nW = int((image_h * sin) + (image_w * cos))
            #nH = int((image_h * cos) + (image_w * sin))

            # adjust the rotation matrix to take into account translation
            #M[0, 2] += (nW / 2) - cX
            #M[1, 2] += (nH / 2) - cY


            # rotate image
            img = cv2.warpAffine(img, M, img.shape[:2])

            # rotate points
            # method 1
            top = M.dot(np.array((top_x, top_y, 1)))
            top_x, top_y = int(top[0]), int(top[1])
            left = M.dot(np.array((left_x, left_y, 1)))
            left_x, left_y = int(left[0]), int(left[1])
            right = M.dot(np.array((right_x, right_y, 1)))
            right_x, right_y = int(right[0]), int(right[1])
            # method 2
            #radian = math.radians(angle)
            #top_x = int(cX + math.cos(radian) * (top_x - cX) - math.sin(radian) * (top_y - cY))
            #top_y = int(cY + math.sin(radian) * (top_x - cX) + math.cos(radian) * (top_y - cY))


        # crop around new rotated top point
        crop_offset = 0
        x = max(0, top_x - self.crop_size//2 + random.randint(-crop_offset,crop_offset))
        y = max(0, top_y - self.crop_size//2 + random.randint(-crop_offset,crop_offset))
        x = min(x, self.image_h - self.crop_size)
        y = min(y, self.image_h - self.crop_size)
        crop = img[y:y+self.crop_size, x:x+self.crop_size]

        # move points to crop
        top_x, top_y = top_x-x, top_y-y
        left_x, left_y = left_x-x, left_y-y
        right_x, right_y = right_x-x, right_y-y

        if self.show:
            # draw new top point in crop
            crop = cv2.circle(crop, (top_x, top_y), 2, (0,0,255), 3)
            crop = cv2.circle(crop, (left_x, left_y), 2, (0,255,0), 3)
            crop = cv2.circle(crop, (right_x, right_y), 2, (0,0,255), 3)

        #left_x, left_y = int(values[2]*image_w), int(values[3]*image_h)
        #crop = cv2.circle(crop, (left_x-x, left_y-y), 2, (0,0,255), 3)

        #right_x, right_y = int(values[4]*image_w), int(values[5]*image_h)
        #crop = cv2.circle(crop, (right_x-x, right_y-y), 2, (0,0,255), 3)

        #line_l = points2line([top_x/self.image_w, top_y/self.image_h],
        #                     [left_x/self.image_w, left_y/self.image_h])

        #line_r = points2line([top_x/self.image_w, top_y/self.image_h],
        #                     [right_x/self.image_w, right_y/self.image_h])


        t = np.array([top_x/self.crop_size, top_y/self.crop_size])
        l = np.array([left_x/self.crop_size, left_y/self.crop_size])
        r = np.array([right_x/self.crop_size, right_y/self.crop_size])

        v_l = t - l
        v_l = (v_l/np.linalg.norm(v_l))
        v_r = t - r
        v_r = (v_r/np.linalg.norm(v_r))
        v_m = (v_l+v_r)/2
        v_m = v_m/np.linalg.norm(v_m)
        if self.show:
            l = t - v_l
            left_x, left_y = int(l[0]*self.crop_size), int(l[1]*self.crop_size)
            crop = cv2.line(crop, (top_x, top_y), (left_x, left_y), (255,0,0), 1)
            r = t - v_r
            right_x, right_y = int(r[0]*self.crop_size), int(r[1]*self.crop_size)
            crop = cv2.line(crop, (top_x, top_y), (right_x, right_y), (255,0,0), 1)
            #crop = cv2.circle(crop, (right_x, right_y), 2, (0,0,255), 3)

            m = t - (v_m)
            middle_x_r, middle_y_r = int(m[0]*self.crop_size), int(m[1]*self.crop_size)
            crop = cv2.line(crop, (top_x, top_y), (middle_x_r, middle_y_r), (255,255,0), 1)

            cv2.imshow("crop", crop)
            cv2.waitKey()

        return crop, torch.Tensor([t[0], t[1], v_m[0], v_m[1]]), (x,y)#torch.Tensor([v_m[0], v_m[1]])#torch.Tensor([t[0], t[1], l[0], l[1], r[0], r[1]])#torch.Tensor([line_l[0], line_l[1], line_r[0], line_r[1]])
'''
