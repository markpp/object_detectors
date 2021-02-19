import argparse
import os
import yaml
import numpy as np
import torch
import pytorch_lightning as pl

import cv2


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
    ap.add_argument("-d", "--data", type=str,
                    default='/home/markpp/datasets/teejet/iphone_data/train_list.txt', help="path to list of files")
    args = vars(ap.parse_args())

    #import sys
    #sys.path.append('../pytorch/')
    #from datasets.doors.datamodule import DoorDataModule
    with torch.no_grad():

        from MaskRCNN import create_model
        model = create_model(num_classes=2,pretrained=False)#.load_from_checkpoint(args['checkpoint'])

        # initialize state_dict from checkpoint to model
        model.load_state_dict(torch.load(args['checkpoint']))
        model.eval()

        from torchsummary import summary
        #summary(model, input_size=(3, 800, 800), device='cpu')
        #summary(model,  device='cpu')
        #torch.rand((3, 800, 800))

        filename = "aligned_out_item19_image74_2197_4655_3221_5679.jpg"
        #filename = "2007_000423.jpg"

        input = cv2.imread(filename)
        input = cv2.resize(input, (800, 800))
        img = input.copy()
        img = img.transpose((2, 0, 1))
        img = img / 255.0
        img[0] = (img[0] - 0.485)/0.229
        img[1] = (img[1] - 0.456)/0.224
        img[2] = (img[2] - 0.406)/0.225
        img = torch.from_numpy(img)
        img = img.float()
        img = img.unsqueeze(0)

        print(img.shape)
        out = model(img)[0]

        boxes, labels, scores, masks = out['boxes'].numpy(), out['labels'].numpy(), out['scores'].numpy(), out['masks'].mul(255).byte().numpy()

        best_idx = np.argmax(scores)

        #box = boxes[best_idx]
        #print(best_idx)

        mask = np.zeros(input.shape[:2], dtype=np.uint8)

        for idx, (box, score) in enumerate(zip(boxes,scores)):
            #print(masks[idx][0].shape)
            #print(np.where(masks[idx][0]>0))
            #print(masks[idx,0].max())
            #print(masks[idx,0].min())
            if score > 0.1:
                print(score)
                x1, y1, x2, y2 = map(int, box)
                #print([x1, y1, x2, y2])
                input = cv2.rectangle(input,(x1, y1),(x2, y2),(0,255,0),2)
                cv2.putText(input, "{:.2f}".format(score), (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                indices = np.where(masks[idx][0]>mask)
                mask[indices] = masks[idx][0][indices]

            #cv2.imshow("mask",masks[idx][0])
            #cv2.waitKey()


        #
        #for m in masks:
        #    indices = np.where(m>0)
        #    mask[indices] = m[indices]


        #.detach()
        cv2.imshow("input",input)
        cv2.imshow("mask",mask)
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
