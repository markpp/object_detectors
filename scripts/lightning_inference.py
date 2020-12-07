import argparse
import os
import yaml

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
                    default='trained_models/model.pkl', help="path to the checkpoint file")
    ap.add_argument("-d", "--data", type=str,
                    default='/home/markpp/datasets/teejet/iphone_data/train_list.txt', help="path to list of files")
    args = vars(ap.parse_args())

    import sys
    sys.path.append('../pytorch')
    from datasets.doors.datamodule import DoorDataModule
    sys.path.append('../pytorch/faster_rcnn')
    from faster_rcnn import FasterRCNN
    model = FasterRCNN(num_classes=2).load_from_checkpoint(args['checkpoint'])
    #model.load_state_dict(torch.load(args['checkpoint']))
    #model = torch.load(args['checkpoint'])
    #print(model)
    model.eval()

    input = cv2.imread("A1326PE_2_id-000000.png",0)
    img = input.copy()
    #img = img.transpose((2, 0, 1))
    img = img / 255.0
    img = torch.from_numpy(img)
    img = img.float()
    img = img.unsqueeze(0)

    print(img.shape)
    out = model(img.unsqueeze(0))
    print(out)

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
