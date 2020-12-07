from engine import train_one_epoch, evaluate
#from torchvision import transforms as T
from custom_transform import Compose, RandomAffine, RandomHorizontalFlip, RandomCrop, ToTensor

def get_transform(train):
    tfms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    if train:
        tfms.append(RandomAffine(degrees=90, translate=(0.2,0.2), scale=(0.8,1.2), shear=25, resample=False, fillcolor=0))
    tfms.append(RandomCrop(224))
    if train:
        tfms.append(RandomHorizontalFlip(0.5))
    tfms.append(ToTensor())
    #tfms.append(Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]))
    return Compose(tfms)
'''
# helper functions for data augmentation / transformation, which leverages the functions in refereces/detection
import transforms as T
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        #transforms.append(T.RandomAffine(degrees=90, translate=(0.2,0.2), scale=(0.8,1.2), shear=25, resample=False, fillcolor=0))
        #transforms.append(T.RandomCrop(512, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'))
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
'''

import torch
import utils
from data_loader import CustomDataset, remove_samples_without_annotations
# use our dataset and defined transformations
dataset_val = remove_samples_without_annotations(CustomDataset('/home/markpp/datasets/bo/val_1024', get_transform(train=False)))
dataset_train = remove_samples_without_annotations(CustomDataset('/home/markpp/datasets/bo/val_1024', get_transform(train=False)))#remove_samples_without_annotations(CustomDataset('/home/markpp/datasets/bo/train_1024', get_transform(train=True)))


# define training and validation data loaders
data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=4, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=4, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)


# lets see what we got assigned
print('PyTorch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

from model import get_instance_segmentation_model
# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
model.to(device) # move model to the right device

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# let's train it for 10 epochs
num_epochs = 20

from model import train_one_epoch_
for epoch in range(num_epochs):

    # train for one epoch, printing every 10 iterations
    #train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    train_one_epoch_(model, optimizer, data_loader_train, device, epoch, print_freq=10)

    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_val, device=device)

torch.save(model,'final_model.pkl')
