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
    ap.add_argument("-i", "--image", type=str,
                    default='A1326PE_2_id-000000.png', help="path to image file")
    args = vars(ap.parse_args())
    #L1415_H1_E1_K13_id-000014.png
    #L1415_H1_E1_K13_id-000014.png
    #A1326PE_2_id-000000.png
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
        #from torchsummary import summary
        #summary(model, (3, 800, 800))
        # optionally, if you want to export the model to ONNX:
        #x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        #torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)


        #input = cv2.imread(filename,0)

        input = cv2.imread(args['image'])
        img = input.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32) / 255.0
        #img[0] = (img[0] - 0.485)/0.229
        #img[1] = (img[1] - 0.456)/0.224
        #img[2] = (img[2] - 0.406)/0.225
        img[0] = (img[0] - 0.5)/0.5
        img[1] = (img[1] - 0.5)/0.5
        img[2] = (img[2] - 0.5)/0.5
        img = torch.from_numpy(img)
        #img = img.float()
        #img = img.unsqueeze(0)

        print(img.shape)
        out = model(img.unsqueeze(0))[0]

        boxes, labels, scores = out['boxes'].numpy(), out['labels'].numpy(), out['scores'].numpy()

        nms_boxes, nms_scores = [], []
        for box,score in zip(boxes,scores):
            #x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = box
            nms_boxes.append([x1,y1,x2-x1,y2-y1])
            nms_scores.append(score)

        from nms import nms
        #nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms
        indicies =  nms.boxes(nms_boxes, nms_scores, nms_function=nms.malisiewicz.nms, nms_threshold=0.1)
        boxes = boxes[indicies]
        scores = scores[indicies]


        #best_idx = np.argmax(scores)
        #box = boxes[best_idx]
        #print(best_idx)
        for box,score in zip(boxes,scores):
            #print(box)
            print(score)
            if score > 0.5:
                x1, y1, x2, y2 = map(int, box)
                #print([x1, y1, x2, y2])
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
