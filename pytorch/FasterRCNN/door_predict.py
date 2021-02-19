import torch
import numpy as np
import numpy as np
import os

import cv2
from nms import nms

if __name__ == "__main__":
    crop_size = 800
    root_dir = '/home/markpp/github/MapGeneralization/data/Public'
    list = 'train_list_reduced_cnn.txt'

    # load model
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
        model.load_state_dict(torch.load('trained_models/model.pt'))
        model.eval()



        # load data
        with open(os.path.join(root_dir,list)) as f:
            lines = f.read().splitlines()

        img_list = []
        for path in lines:
            img_list.append(os.path.join(root_dir,path))

        for img_path in img_list[:]:
            input = cv2.imread(img_path)

            pred_path = img_path.replace('images','rcnn_bboxes').replace('png','txt')
            pred_file = open(pred_path, 'w')

            img = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
            (T, img) = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            #img = cv2.cvtColor(), cv2.COLOR_BGR2RGB)
            img_h, img_w = img.shape[:2]

            # (1) pad the image such that it is divisable by crop_size
            right_pad = crop_size - (img_w % crop_size)
            bottom_pad = crop_size - (img_h % crop_size)
            img = cv2.copyMakeBorder(img, top=0, bottom=bottom_pad, left=0, right=right_pad, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))


            # standard grid crop
            offsets,batch = [],[]
            for x in range(0,img_w,crop_size):
                for y in range(0,img_h,crop_size):
                    #print("x {}, y {}".format(x,y))
                    batch.append(img[y:y+crop_size, x:x+crop_size])
                    offsets.append([x,y])

            # offset standard grid crop
            half = crop_size//2
            for x in range(half,img_w-half,crop_size):
                for y in range(half,img_h-half,crop_size):
                    #print("x {}, y {}".format(x,y))
                    batch.append(img[y:y+crop_size, x:x+crop_size])
                    offsets.append([x,y])

            #print(len(batch))
            #print(offsets)

            boxes, labels, scores = [], [], []

            for offset, crop in zip(offsets,batch):
                x_off, y_off = offset

                crop = crop.transpose((2, 0, 1))
                crop = crop.astype(np.float32) / 255.0
                crop = torch.from_numpy(crop)

                out = model(crop.unsqueeze(0))[0]

                crop_boxes, crop_labels, crop_scores = out['boxes'].detach().numpy(), out['labels'].detach().numpy(), out['scores'].detach().numpy()

                for box,label,score in zip(crop_boxes,crop_labels,crop_scores):
                    if score > 0.5:
                        x1, y1, x2, y2 = map(int, box)
                        x, y, w, h = x1 + x_off, y1 + y_off, x2 - x1, y2 - y1
                        if x + w < img_w and y + h < img_h:
                            boxes.append([x, y, w, h])
                            scores.append(score)
                            labels.append(label)

            #nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms
            indicies =  nms.boxes(boxes, scores, nms_function=nms.malisiewicz.nms, nms_threshold=0.1)
            boxes = np.array(boxes)[indicies]
            scores = np.array(scores)[indicies]
            labels = np.array(labels)[indicies]

            for box,label,score in zip(boxes,labels,scores):
                x1, y1, w, h = box
                x2, y2 = x1+w, y1+h
                img = cv2.rectangle(img,(x1,y1),(x2, y2),(0,255,0),2)

                #pred_file.write("{} {} {} {}\n".format(x1/img_w, y1/img_h, x2/img_w, y2/img_h))
                pred_file.write("{} {} {} {} {}\n".format(score, x2/img_w, y2/img_h, x1/img_w, y1/img_h))

            output_path = os.path.join("pred",os.path.basename(img_path))
            cv2.imwrite(output_path,img)
            cv2.waitKey()
