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
    'image_size': 512,
    'batch_size': 6,
    'name': 'Spray',
    'device': 'cpu',
}

SPRAY_CLASSES = ['blue']

def create_test_batch(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img)
    #img = img.float()
    img = img.unsqueeze(0)
    print(img.shape)
    return img

def test_net(net, thresh):
    #image = cv2.imread("00150.jpg")[200:200+cfg['image_size'],0:0+cfg['image_size']]
    image = cv2.imread("00001.jpg")[150:150+cfg['image_size'],750:750+cfg['image_size']]
    #cv2.imwrite("test.jpg",image)
    scale = np.array([[image.shape[1], image.shape[0]]])
    input = create_test_batch(image.copy())
    dets = net(input)      # forward pass
    bbox_pred, scores, cls_inds = dets
    print(bbox_pred.shape)

    # map the boxes to origin image scale
    bbox_pred *= scale

    CLASSES = SPRAY_CLASSES
    class_color = [(0,255,0)]
    #print(scores)
    for i, box in enumerate(bbox_pred):
        cls_indx = cls_inds[i]
        x, y = box
        if scores[i] > thresh:
            cv2.circle(image, (int(x), int(y)), 5, class_color[int(cls_indx)], 2)
            mess = '%s: %.3f' % (CLASSES[int(cls_indx)], scores[i])
            cv2.putText(image, mess, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color[int(cls_indx)], 2)
    cv2.imshow('detection', image)
    cv2.waitKey(0)


if __name__ == '__main__':



    # load net
    from centernet import CenterNet
    net = CenterNet(device=cfg['device'], input_size=cfg['image_size'], mode='test', num_classes=cfg['num_classes'])
    net.load_state_dict(torch.load("trained_models/model_dict.pth"))
    net.eval()
    print('Finished loading model!')

    # evaluation

    test_net(net, thresh=0.1)
