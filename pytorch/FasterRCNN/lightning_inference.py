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

        #from faster_rcnn import FasterRCNN
        #model = FasterRCNN(num_classes=2)#.load_from_checkpoint(args['checkpoint'])

        from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn
        model = fasterrcnn_resnet50_fpn(
            num_classes=2,
            pretrained=False,
            pretrained_backbone=False,
            trainable_backbone_layers=False,
        )

        # initialize state_dict from checkpoint to model
        model.load_state_dict(torch.load(args['checkpoint']))
        model.eval()

        from torchsummary import summary
        #summary(model, input_size=(1, 800, 800), device='cpu')
        summary(model,  device='cpu')
        #torch.rand((3, 800, 800))

        #filename = "L1415_H1_E1_K13_id-000014.png"
        filename = "A1326PE_2_id-000000.png"

        input = cv2.imread(filename,0)
        img = input.copy()
        #img = img.transpose((2, 0, 1))
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.float()
        img = img.unsqueeze(0)

        print(img.shape)
        out = model(img.unsqueeze(0))[0]

        boxes, labels, scores = out['boxes'].numpy(), out['labels'].numpy(), out['scores'].numpy()

        #best_idx = np.argmax(scores)
        #box = boxes[best_idx]
        #print(best_idx)
        for box in boxes:
            print(box)

            x1, y1, x2, y2 = map(int, box)
            print([x1, y1, x2, y2])

            input = cv2.rectangle(input,(x1, y1),(x2, y2),(0,255,0),2)

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
