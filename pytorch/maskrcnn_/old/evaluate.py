import os                                                                    
import torch
import torchvision
import cv2
import utils
import numpy as np
import transforms as T
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from engine import evaluate

CUSTOM_TAGS = [
        '__background__', 'defect'
]



def im_to_tensor(img):
    '''
    Translates an image from numpy to tensor

    input
        img: numpy array

    return
        tn: tensor
    '''
    tn = torch.from_numpy(img)
    tn = tn.permute(2,0,1)
    tn = tn.type(torch.float)
    tn = tn / tn.max()
    return tn

def get_prediction(model, img, device, score_threshold=0.5):
    '''
    Object detection on a single image

    input
        model: torchvision.models.detection.xxx()
        img: numpy array
        device: torch.device()
        threshold: float between 0 and 1

    return
        pred_boxes: list of predicted bounding boxes
        pred_class: list of predicted classes
    '''
    tn = im_to_tensor(img)
    tn = tn.to(device) # Use GPU if available
    pred = model([tn]) 

    # Get the Prediction Score
    pred_class = [CUSTOM_TAGS[i] for i in list(pred[0]['labels'].cpu().numpy()) if i < len(CUSTOM_TAGS)] 

    # Bounding boxes
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] 
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    # Masks
    pred_masks = [i.transpose((1,2,0)).reshape(i.shape[1],i.shape[2]) for i in list(pred[0]['masks'].detach().cpu().numpy())]


    # Get list of index with score greater than threshold.
    pred_t = [pred_score.index(x) for x in pred_score if x > score_threshold]
    if len(pred_t) > 0:
        pred_t = pred_t[-1] 
        pred_masks = pred_masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
    else:
        pred_masks = []
        pred_boxes = []
        pred_class = []


    return pred_masks, pred_boxes, pred_class


# def get_data_loader(path):
#     '''
#     Get data_loader

#     input
#         path: path to the root of the datafolder

#     return
#         data_loader: torch.utils.data.Dataloader
#     '''
#     dataset = PennFudanDataset(path, transforms=get_transform(train=False))

#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=1, shuffle=False, num_workers=0,
#         collate_fn=utils.collate_fn)
#     return data_loader

if __name__ == '__main__':
    # Evaluate using either a data_loader (annotated folder of images) or a single image
    data_loader = False
    example = True
    p = "./bangolufsen/train2017"#'/home/anne/Documents/MaskRCNN_anne/mask-rcnn/bangolufsen/val2017'
    score_threshold = 0.5
    mask_threshold = 0.5

    # train on the GPU or on the CPU, if a GPU is not available 
    if torch.cuda.is_available():
        device = torch.device('cuda') 
        print("Using GPU")
    else:
        print("WARNING: Using CPU")
        device = torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    model = torchvision.models.detection.__dict__["maskrcnn_resnet50_fpn"](num_classes=num_classes, pretrained=False)
    model.to(device)
    # weight_path = "../Sessions/2019_12_10-07_46_33/models/Mask_RCNN_resnet50_10epochs.tar"
    weight_path = "./firstgen/model_11.pth"# "/home/anne/Documents/MaskRCNN_anne/mask-rcnn/firstgen/model_11.pth"

    model.load_state_dict(torch.load(weight_path)["model"])
    # model.load_state_dict(torch.load(weight_path)["model_state_dict"])
    model.to(device)
    model.eval()

    # Evaluate on folder
    #if data_loader:
    #    data_loader = get_data_loader(path=p)
        # evaluate on the test dataset
    #    evaluate(model, data_loader, device=device)

    # Draw predicted bb and labels on images
    if example:
        img_paths = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p,f)) and os.path.splitext(f)[1] == '.png']

        # Ensure that the ouput-folder exists
        out_folder = os.path.join(p,'output')
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)

        for f in img_paths:
            head, tail = os.path.split(f)
            name, ext = os.path.splitext(tail)

            # Read image
            img = cv2.imread(os.path.join(p,f))

            # Predict 
            pred_masks, pred_boxes, pred_class = get_prediction(model, img, device,score_threshold=score_threshold)
            
            # Combine masks and img
            for idx in range(len(pred_masks)):
                #Get all the predicted stuff
                mask = pred_masks[idx]
                mask[mask < mask_threshold] = 0

                boxes = pred_boxes[idx]
                boxes = [(int(boxes[0][0]), int(boxes[0][1])), (int(boxes[1][0]), int(boxes[1][1]))]
                
                p_cls = pred_class[idx]

                # Draw semitransparent mask on the image
                bool_mask = mask.astype(np.bool)
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                combined = ((0.4 * np.asarray(color)) + (0.6 * img[bool_mask])).astype(np.uint8)
                img[bool_mask] = combined

                # Add bounding box and predicted class
                cv2.rectangle(img, boxes[0], boxes[1], color,  1) # Draw bounding box
                cv2.putText(img, p_cls, boxes[0],  cv2.FONT_HERSHEY_SIMPLEX, 1, color,1) # Write the predicted class

                # Write mask to folder
                cv2.imwrite(os.path.join(out_folder, name+"_mask_{}".format(idx)+ext), (mask*255).astype(np.uint8))
            
            # Store image using filename of input image
            img_path = os.path.join(out_folder,name+'_out'+ext)
            cv2.imwrite(img_path, img)

            print('\n*************')
            print("The test image '{0}' was processed using the trained model. The output image with bounding box detections and labels can be found in {1}".format(tail,img_path))
            print('\n*************')
