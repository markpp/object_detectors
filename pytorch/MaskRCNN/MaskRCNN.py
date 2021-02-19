import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import numpy as np
import cv2

def create_model(num_classes,pretrained=True):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected
        return torch.tensor(0.0, device=pred["boxes"].device)
    return torchvision.ops.box_iou(target["boxes"], pred["boxes"]).diag().mean()

def _plot_boxes(imgs, targets, preds):
    """
    Plot the target and prediction boxes
    """
    dets = []
    for img, tar, pred in zip(imgs, targets, preds):
        out = img.cpu()
        out[0] = out[0] * 0.229 + 0.485
        out[1] = out[1] * 0.224 + 0.456
        out[2] = out[2] * 0.225 + 0.406
        out = out.mul(255).permute(1, 2, 0).byte().numpy()
        for b,l in zip(tar["boxes"],tar["labels"]):
            x1, y1, x2, y2 = [int(x) for x in b.tolist()]
            cv2.rectangle(out,(x1, y1),(x2, y2),(255,0,0),3)
        for b,l,s in zip(pred["boxes"],pred["labels"],pred["scores"]):
            score = s.item()
            if score > 0.25:
                x1, y1, x2, y2 = [int(x) for x in b.tolist()]
                cv2.rectangle(out,(x1, y1),(x2, y2),(0,0,255),2)
                cv2.putText(out,"{:.2f}".format(score), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        if len(dets):
            dets = np.concatenate((dets, out), axis=1)
        else:
            dets = out
    return dets
