import argparse
import os
import yaml
import numpy as np
import torch
import pytorch_lightning as pl

import cv2




def encode_segmap(mask):
    """
    Sets void classes to zero so they won't be considered for training
    """
    DEFAULT_VOID_LABELS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1)
    DEFAULT_VALID_LABELS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
    ignore_index = 250
    class_map = dict(zip(DEFAULT_VALID_LABELS, range(len(DEFAULT_VALID_LABELS))))

    for voidc in DEFAULT_VOID_LABELS:
        mask[mask == voidc] = ignore_index
    for validc in DEFAULT_VALID_LABELS:
        mask[mask == validc] = class_map[validc]
    # remove extra idxs from updated dataset
    mask[mask > 18] = ignore_index
    return mask


if __name__ == '__main__':
    """
    Apply detector model to RGB image.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    ap.add_argument("-i", "--image", type=str,
                    default='training/image_2/000000_10.png', help="path image file")
    args = vars(ap.parse_args())

    #import sys
    #sys.path.append('../pytorch/')
    #from datasets.doors.datamodule import DoorDataModule
    with torch.no_grad():

        '''
        from lightning_model import SemSegment
        model = SemSegment(datamodule=dm,
                           lr = 0.01,
                           num_classes = 19,
                           num_layers = 5,
                           features_start = 64,
                           bilinear = False)
        '''
        from unet import UNet
        model = UNet(num_classes=19,
                     num_layers=5,
                     features_start=64,
                     bilinear=False)

        #model = torch.load(args['checkpoint'])
        # load check point
        checkpoint = torch.load(args['checkpoint'])
        # initialize state_dict from checkpoint to model
        #net.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint)

        model.eval()

        #from torchsummary import summary
        #summary(model, input_size=(3, 1242, 375))

        #filename = "testing/image_2/000000_10.png"
        filename = "training/image_2/000000_10.png"

        input = cv2.imread(args['image'])
        img = input.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = img / 255.0
        img[0] = (img[0]-0.35675976)/0.32064945
        img[1] = (img[1]-0.37380189)/0.32098866
        img[2] = (img[2]-0.3764753)/0.32325324
        img = torch.from_numpy(img)
        img = img.float()

        out = model(img.unsqueeze(0))[0]

        import random
        hex_colors = ['#'+''.join(random.choice('0123456789ABCDEF') for n in range(6)) for i in range(19)]
        from PIL import ImageColor
        bgr_colors = [ImageColor.getrgb(c) for c in hex_colors]

        indicies = np.argmax(out, axis=0)
        map = input.copy()
        for r in range(map.shape[0]):
            for c in range(map.shape[1]):
                map[r,c] = bgr_colors[indicies[r,c]]

        cv2.imshow("map",map)


        filename = "training/semantic/000000_10.png"
        gt = cv2.imread(filename,-1)
        gt = encode_segmap(gt)
        gt_map = input.copy()

        for r in range(gt_map.shape[0]):
            for c in range(gt_map.shape[1]):
                color_idx = gt[r,c]
                if color_idx < len(bgr_colors):
                    gt_map[r,c] = bgr_colors[color_idx]
                else:
                    gt_map[r,c] = (0,0,0)

        cv2.imshow("gt_map",gt_map)


        '''
        for o in out:
            o = torch.sigmoid(o)
            #print(o.numpy().max())
            output = o.mul(255).byte().numpy()
            cv2.imshow("car",output)
            cv2.waitKey(500)
        '''
        '''
        boxes, labels, scores = out['boxes'].numpy(), out['labels'].numpy(), out['scores'].numpy()

        best_idx = np.argmax(scores)
        box = boxes[best_idx]
        print(best_idx)
        print(box)
        x1, y1, x2, y2 = box
        #img = cv2.rectangle(img,(x1, y1),(x2, y2),(0,255,0),2)
        '''
        #.detach()
        cv2.imshow("input",input)
        cv2.waitKey()
        '''
        #model = LightningAutoencoder(config).load_from_checkpoint(args['checkpoint'])


        dataset = SprayDataset(args['data'], crop_size=config['exp_params']['img_size'])
        input = dataset[100][0]

        rec, a, b, c = model(input.unsqueeze(0))

        print(rec.shape)

        from torchvision.transforms.transforms import Normalize

        #unnormalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
        #rec = unnormalize(rec[0])
        rec = rec[0]
        rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
        cv2.imwrite("rec.png",rec)

        #input = unnormalize(input)
        input = input.mul(255).permute(1, 2, 0).byte().numpy()
        cv2.imwrite("input.png",cv2.cvtColor(input, cv2.COLOR_RGB2BGR))
        '''
