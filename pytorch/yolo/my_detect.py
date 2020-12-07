import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

model = 0
device = 0

def mask2gt(mask):
    # instances are encoded as different colors
    obj_ids = np.unique(mask)

    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]

    boxes = []
    # if no objects exists
    if len(obj_ids) == 0:
        return boxes

    for i, obj in enumerate(obj_ids):
        pos = np.where(masks[i])
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        ymin, ymax = np.min(pos[0]), np.max(pos[0])
        boxes.append([float(xmin)/mask.shape[1], float(ymin)/mask.shape[0],
                      float(xmax)/mask.shape[1], float(ymax)/mask.shape[0]])
    return boxes

def label_unique(mask):
    new_mask = np.zeros(mask.shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i, cont in enumerate(contours):
        cv2.drawContours(new_mask, [cont], contourIdx=-1, color=(i+1), thickness=-1)
    return new_mask

def detect(batch, offsets):

    img = torch.from_numpy(batch).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    #img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    confs, bbs = [], []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            off = offsets[i]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (1024,1024,3)).round()

            for d in det:
                x, y, x_, y_, c = int(d[0]), int(d[1]), int(d[2]), int(d[3]), float(d[4])
                bbs.append([x+off[0], y+off[1], x_-x, y_-y])
                confs.append(x)
    return confs, bbs

def data_loader(crop_size = 1024, root_dir = '/home/markpp/datasets/bo/test/', version='mixed', make_gt=False):

    n_width, n_height = (3300+crop_size)//crop_size, (6050+crop_size)//crop_size
    batch_size = n_width * n_height
    print(batch_size)

    items = sorted([f for f in os.listdir(root_dir) if 'item' in f])
    # iterate over each item
    for item in items[:]:
        img_paths = sorted([f for f in os.listdir(os.path.join(root_dir,item,"rgb")) if f.endswith('.jpg')])
        print("{} has {} images".format(item,len(img_paths)))
        # iterate over each image belonging to the given item
        if not os.path.exists(os.path.join(root_dir,item,"results_{}".format(version))):
            os.makedirs(os.path.join(root_dir,item,"results_{}".format(version)))
        if not os.path.exists(os.path.join(root_dir,item,"pred_{}".format(version))):
            os.makedirs(os.path.join(root_dir,item,"pred_{}".format(version)))

        for i,img_file in enumerate(img_paths[:]):
            start = time.time()

            pred_bb, pred_cls, pred_conf = [], [], []
            gt_bb, gt_cls = [], []

            img = cv2.imread(os.path.join(root_dir,item,"rgb",img_file))
            img = cv2.copyMakeBorder(img, top=0, bottom=crop_size, left=0, right=crop_size, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                print("image {} not loaded".format(img_file))
                continue

            if make_gt:
                mask_file = img_file.replace('jpg','png')
                mask = cv2.imread(os.path.join(root_dir,item, "masks",mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
                    print("empty mask created in place of missing file {}".format(mask_file))
                else:
                    mask = cv2.copyMakeBorder(mask, top=0, bottom=crop_size, left=0, right=crop_size, borderType=cv2.BORDER_CONSTANT, value=0)


                new_mask  = label_unique(mask)
                #cv2.imwrite(os.path.join(root_dir,item,"results",mask_file),new_mask)

                gts = mask2gt(new_mask)
                for gt in gts:
                    gt_bb.append(gt) #xmin, ymin, xmax, ymax
                    gt_cls.append(0)

            offsets = []
            batch = np.empty((batch_size,3,crop_size,crop_size))
            idx = 0
            # divide image into appropritate crops and detect defects
            for x in range(0,img.shape[1]-crop_size,crop_size):
                for y in range(0,img.shape[0]-crop_size,crop_size):
                    crop = img[y:y+crop_size, x:x+crop_size]
                    batch[idx] = np.rollaxis(crop, 2, 0) / 255
                    idx += 1
                    offsets.append([x,y])
                    if idx == batch_size:
                        #detect(batch, offsets)

                        confs, bbs = detect(batch, offsets)

                        for conf, bb in zip(confs, bbs):
                            left, top, width, height = bb[0], bb[1], bb[2], bb[3] # add crop offset
                            img = cv2.rectangle(img, (left, top), (left + width, top + height), (0,255,0), 2)
                            cv2.putText(img, "{:.2f}".format(conf), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                            pred_bb.append([float(left)/img.shape[1], float(top)/img.shape[0],
                                            float(left+width)/img.shape[1], float(top+height)/img.shape[0]])
                            pred_cls.append(0)
                            pred_conf.append(conf)
                        batch = []
                        offsets = []

            cv2.imwrite(os.path.join(root_dir,item,"results_{}".format(version),img_file),img)

            pred_name = img_file.replace('jpg','txt')
            pred_path = os.path.join(root_dir,item,"pred_{}".format(version),pred_name) #os.path.join(root_dir,item,"pred",pred_name)
            if len(pred_bb):
                with open(pred_path, 'w') as pred_file:
                    for conf, bb in zip(pred_conf,pred_bb):
                        pred_file.write('defect {:.3f} {} {} {} {}\n'.format(conf,
                                                                         int(bb[0]*img.shape[1]),
                                                                         int(bb[1]*img.shape[0]),
                                                                         int(bb[2]*img.shape[1]),
                                                                         int(bb[3]*img.shape[0])))
            else:
                np.savetxt(pred_path, [], delimiter=",", fmt='%u')

            if make_gt:
                gt_name = img_file.replace('jpg','txt')
                gt_path = os.path.join(root_dir,item,"gt",gt_name) #os.path.join(root_dir,item,"gt",gt_name)
                if len(gt_bb):
                    with open(gt_path, 'w') as gt_file:
                        for bb in gt_bb:
                            gt_file.write('defect {} {} {} {}\n'.format(int(bb[0]*img.shape[1]),
                                                                        int(bb[1]*img.shape[0]),
                                                                        int(bb[2]*img.shape[1]),
                                                                        int(bb[3]*img.shape[0])))
                else:
                    np.savetxt(gt_path, [], delimiter=",", fmt='%u')

            end = time.time()
            print("{} took {}".format(img_file, end - start))

#python3 my_detect.py --weights best.pt --img-size 1024


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():

        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(opt.weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                data_loader(crop_size = opt.img_size)
                #detect()
                strip_optimizer(opt.weights)
        else:
            data_loader(crop_size = opt.img_size)
            #detect()
