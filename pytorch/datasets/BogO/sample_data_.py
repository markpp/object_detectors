
import numpy as np
import os
import cv2
from mean_average_precision.detection_map import DetectionMAP
from mean_average_precision.utils.show_frame import show_frame
import time
from image_patcher import crop_patches
from onnx_predict import YOLO

PREDICT = False

if __name__ == "__main__":

    crop_size = 608

    #n_w, n_h = ((3300-crop_size)//crop_size), (6050//crop_size)
    n_w, n_h = ((3300+crop_size)//crop_size), ((6050+crop_size)//crop_size)
    batch_size = n_w * n_h

    if PREDICT:
        yolo = YOLO(model_path="yolov3-spp-{}_608.onnx".format(batch_size))

    root_dir = '/home/markpp/datasets/bo/train/'
    #items = ['item00', 'item01', 'item02', 'item03', 'item04', 'item07', 'item08']
    items = sorted([f for f in os.listdir(root_dir) if 'item' in f])

    # iterate over each item
    for item in items[:]:
        img_paths = sorted([f for f in os.listdir(os.path.join(root_dir,item,"rgb")) if f.endswith('.jpg')])
        print("{} has {} images".format(item,len(img_paths)))
        # iterate over each image belonging to the given item
        if not os.path.exists(os.path.join(root_dir,item,"results")):
            os.makedirs(os.path.join(root_dir,item,"results"))
        if not os.path.exists(os.path.join(root_dir,item,"gt")):
            os.makedirs(os.path.join(root_dir,item,"gt"))
        if not os.path.exists(os.path.join(root_dir,item,"pred")):
            os.makedirs(os.path.join(root_dir,item,"pred"))

        for i,img_file in enumerate(img_paths[:]):
            start = time.time()

            img = cv2.imread(os.path.join(root_dir,item,"rgb",img_file))
            img = cv2.copyMakeBorder(img, top=0, bottom=crop_size, left=0, right=crop_size, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
            img_h, img_w = img.shape[:2]

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                print("image {} not loaded".format(img_file))
                break

            patches, patch_offsets, img_gt_bb, img_gt_cls = crop_patches(img, crop_size, root_dir, item, img_file)

            img_pred_bb, img_pred_cls, img_pred_conf = [], [], []

            if len(patches) == batch_size:
                if PREDICT:
                    confs, bbs = yolo.predict(np.array(patches, dtype=np.float32), patch_offsets)

                for bb, cl in zip(img_gt_bb, img_gt_cls):
                    left, top, right, bottom = int(bb[0]*img_w), int(bb[1]*img_h), int(bb[2]*img_w), int(bb[3]*img_h)
                    img = cv2.rectangle(img, (left, top), (right, bottom), (255,0,0), 2)

                if PREDICT:
                    for conf, bb in zip(confs, bbs):
                        left, top, width, height = bb[0], bb[1], bb[2], bb[3]
                        img = cv2.rectangle(img, (left, top), (left + width, top + height), (0,255,0), 2)
                        cv2.putText(img, "{:.2f}".format(conf), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                        img_pred_bb.append([float(left)/img.shape[1], float(top)/img.shape[0],
                                            float(left+width)/img.shape[1], float(top+height)/img.shape[0]])
                        img_pred_cls.append(0)
                        img_pred_conf.append(conf)

                cv2.imwrite(os.path.join(root_dir,item,"results",img_file),img)

                if PREDICT:
                    pred_name = img_file.replace('jpg','txt')
                    pred_path = os.path.join(root_dir,item,"pred",pred_name) #os.path.join(root_dir,item,"pred",pred_name)
                    if len(img_pred_bb):
                        with open(pred_path, 'w') as pred_file:
                            for conf, bb in zip(img_pred_conf,img_pred_bb):
                                pred_file.write('defect {:.2f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(conf,
                                                                                 float(bb[0]),
                                                                                 float(bb[1]),
                                                                                 float(bb[2]),
                                                                                 float(bb[3])))
                    else:
                        np.savetxt(pred_path, [], delimiter=",", fmt='%u')

                gt_name = img_file.replace('jpg','txt')
                gt_path = os.path.join(root_dir,item,"gt",gt_name) #os.path.join(root_dir,item,"gt",gt_name)
                if len(img_gt_bb):
                    with open(gt_path, 'w') as gt_file:
                        for bb in img_gt_bb:
                            gt_file.write('defect {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(float(bb[0]),
                                                                        float(bb[1]),
                                                                        float(bb[2]),
                                                                        float(bb[3])))
                else:
                    np.savetxt(gt_path, [], delimiter=",", fmt='%u')

                end = time.time()
                print("{} took {}".format(img_file, end - start))

    '''
    import matplotlib.pyplot as plt

    frames = np.load("frames.npy", allow_pickle=True)
    print(frames[0])

    mAP = DetectionMAP(n_class=1)
    for i, frame in enumerate(frames):
        print("Evaluate frame {}".format(i))
        #show_frame(*frame)
        mAP.evaluate(*frame)

    mAP.plot()
    plt.show()
    '''
