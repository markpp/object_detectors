import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import tools
import time


cfg = {
    'num_classes': 1,
    'lr_epoch': (30, 40),
    'max_epoch': 50,
    'image_size': 600,
    'name': 'Spray',
}

SPRAY_CLASSES = ['blue']

def test_net(net, thresh):

    input = cv2.imread("00001.jpg")[200:1000,200:1000]
    img = input.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32) / 255.0
    #img = img.transpose((2, 0, 1))
    #img = torch.from_numpy(img)/255.0
    img = torch.from_numpy(img)
    #img = img.float()
    #img = img.unsqueeze(0)
    print(img.shape)

    dets = net(img.unsqueeze(0))      # forward pass

    scale = np.array([[input.shape[1], input.shape[0],
                       input.shape[1], input.shape[0]]])

    bbox_pred, scores, cls_inds = dets

    # map the boxes to origin image scale
    bbox_pred *= scale

    CLASSES = SPRAY_CLASSES
    class_color = tools.CLASS_COLOR
    for i, box in enumerate(bbox_pred):
        cls_indx = cls_inds[i]
        xmin, ymin, xmax, ymax = box
        if scores[i] > thresh:
            box_w = int(xmax - xmin)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
            cv2.rectangle(img, (int(xmin), int(abs(ymin)-15)), (int(xmin+box_w*0.55), int(ymin)), class_color[int(cls_indx)], -1)
            mess = '%s: %.3f' % (CLASSES[int(cls_indx)], scores[i])
            cv2.putText(img, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv2.imshow('detection', img)
    cv2.waitKey(0)


if __name__ == '__main__':



    # load net
    from centernet import CenterNet
    net = CenterNet(input_size=cfg['image_size'], num_classes=cfg['num_classes'])
    net.load_state_dict(torch.load("trained_models/model_dict_.pth"))
    net.eval()
    print('Finished loading model!')

    # evaluation

    test_net(net, thresh=0.1)
