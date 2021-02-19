import argparse
import torch
import onnx
import os, pickle
import numpy as np

from torch import nn
from torch.onnx import OperatorExportTypes
import cv2

cfg = {
    'num_classes': 1,
    'lr_epoch': (30, 40),
    'max_epoch': 50,
    'image_size': 512,
    'batch_size': 6,
    'name': 'Spray',
    'device': 'cpu',
}

def create_test_batch(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img)
    #img = img.float()
    img = img.unsqueeze(0)
    print(img.shape)
    return img

def save_model_output_as_json( fn_output, model_outputs ):
    import json
    print( 'torch_outputs: ', model_outputs[0].shape )

    output_data = [
        model_outputs[0][0,:].tolist(), # hmap, save only class 14 [1, 20, 96, 96]
        model_outputs[1][0,:].tolist(), # regs [1, 2, 96, 96]
        model_outputs[2][0,:].tolist(), # w_h_ [1, 2, 96, 96]
    ]
    with open( fn_output, 'w' ) as fp:
        json.dump( output_data, fp )

def show_output(output):
    cls_maps, txty_maps = output
    
    print(txty_maps.shape)

    HW = cls_maps.shape[1]
    h = int(np.sqrt(HW))

    class_heatmap = cls_maps[0].detach().numpy().reshape(h, h)
    class_heatmap = cv2.resize(class_heatmap, (512, 512))
    cv2.imshow("class_heatmap", class_heatmap)
    x_heatmap = txty_maps[0, :, 0].detach().numpy().reshape(h, h)
    x_heatmap = cv2.resize(x_heatmap, (512, 512))
    cv2.imshow("x_heatmap", x_heatmap)
    y_heatmap = txty_maps[0, :, 1].detach().numpy().reshape(h, h)
    y_heatmap = cv2.resize(y_heatmap, (512, 512))
    cv2.imshow("y_heatmap", y_heatmap)
    cv2.waitKey(0)

if __name__ == '__main__':
    # load net
    from centernet import CenterNet
    model = CenterNet(device=cfg['device'], input_size=cfg['image_size'], mode='export', num_classes=cfg['num_classes'])
    model.load_state_dict(torch.load("trained_models/model_dict.pth"))
    model.eval()

    image = cv2.imread("00001.jpg")[100:100+cfg['image_size'],700:700+cfg['image_size']]
    scale = np.array([[image.shape[1], image.shape[0]]])
    img_batch = create_test_batch(image)

    torch_outputs = model(img_batch)
    show_output(torch_outputs)

    input_names = ["input.1"]
    output_names = [ 'output_hmap', 'output_regs']
    #output_names = [ 'output_hmap', 'output_regs', 'output_w_h_']
    #output_names = [ 'bbox_pred', 'scores', 'cls_inds']

    torch.onnx.export(
        model,
        img_batch,
        'trained_models/onnx/model.onnx',
        input_names=input_names,
        output_names=output_names,
        operator_export_type=OperatorExportTypes.ONNX
    )

    fn_output = 'trained_models/onnx/test_image_outputs.json'
    save_model_output_as_json( fn_output, torch_outputs[0] )

# pascal model for validation
# python scripts/export_onnx.py --arch mobilenetv2 --img_size 384 --model_path ckpt/pascal_mobilenetv2_384_dp/checkpoint.t7 --num_classes 20  --output_name /tmp/mobilenetv2_hc64.onnx
# python scripts/export_onnx.py --arch resnet_18 --img_size 384 --model_path ckpt/pascal_resnet18_384_dp/checkpoint.t7 --num_classes 20  --output_name /tmp/resnet_hc64.onnx
