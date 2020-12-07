import os
import argparse
import numpy as np
import csv
import cv2

img_w = 0
img_h = 0


def relativ2pixel(detection, frameHeight, frameWidth):
    center_x, center_y = int(detection[0] * frameWidth), int(detection[1] * frameHeight)
    width, height = int(detection[2] * frameWidth), int(detection[3] * frameHeight)
    left, top = int(center_x - width / 2), int(center_y - height / 2)
    return [left, top, width, height]

def get_bbs_from_file(path):
    boxes_file = open(path,"r")
    bb_lines = boxes_file.readlines()
    bbs = []
    for bb_line in bb_lines:
        x1, y1, x2, y2 = bb_line.split(' ')
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        bbs.append([x1, y1, x2-x1, y2-y1])
    return bbs

def map_bbs_to_img(img, bbs):
    for bb in bbs:
        h_pixels, w_pixels = img.shape[:2]
        x1, y1, x2, y2 = int(bb[0]*w_pixels), int(bb[1]*h_pixels), int((bb[0]+bb[2])*w_pixels), int((bb[1]+bb[3])*h_pixels)
        img = cv2.rectangle(img,(x1, y1),(x2, y2),(0,255,0),2)
    return img

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim), 1/r

if __name__ == "__main__":
    """

    Command:
        python show_yolo.py -g
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gt", type=str,
                    help="Path to gt bb .txt")
    args = vars(ap.parse_args())

    img_path = args["gt"].replace("txt", "png")
    img = cv2.imread(img_path,-1)
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    # start a new yolo txt file with name of image

    boxes = get_bbs_from_file(args["gt"])
    img = map_bbs_to_img(img, boxes)

    '''
    if img.shape[0] > img.shape[1]:
        img, _ = ResizeWithAspectRatio(img, height=1400)
    else:
        img, _ = ResizeWithAspectRatio(img, width=1400)
    '''
    '''
    print(img.shape)
    img_h, img_w = img.shape[1], img.shape[0]

    boxes = []
    lines = []
    with open(args["gt"]) as f:
        lines = f.read().splitlines()
    for line in lines:
        cl, c_x, c_y, w, h = line.split(' ')
        boxes.append(relativ2pixel([float(c_x), float(c_y), float(w), float(h)], img_w, img_h))

    for box in boxes:
        print(box)
        cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (0,255,0), 1)

    '''

    cv2.putText(img, os.path.basename(img_path), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("output",img[-400:,:])
    key = cv2.waitKey()
