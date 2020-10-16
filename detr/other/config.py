
# Coefficients:   width,depth,res,dropout
#'efficientnet-b0': (1.0, 1.0, 224, 0.2),'efficientnet-b1': (1.0, 1.1, 240, 0.2),
#'efficientnet-b2': (1.1, 1.2, 260, 0.3),'efficientnet-b3': (1.2, 1.4, 300, 0.3),
#'efficientnet-b4': (1.4, 1.8, 380, 0.4),'efficientnet-b5': (1.6, 2.2, 456, 0.4),
#'efficientnet-b6': (1.8, 2.6, 528, 0.5),'efficientnet-b7': (2.0, 3.1, 600, 0.5),


#network_architecture = 'resnet18'

from argparse import Namespace
hparams = Namespace(**{'backbone': 'detr_resnet50',
                       'pretrained': True,
                       'model_save_path': 'trained_models/model.pt',
                       'data_path': '/home/markpp/datasets/teejet/iphone_data/',
                       'train_list': 'train_list.txt',
                       'val_list': 'val_list.txt',
                       'input_size': 1024,
                       'n_queries': 100, # gets random nans if changed, other values might not be compatible with pretrained
                       'n_classes': 1,
                       'n_epochs': 5,
                       'detection_threshold': 0.25,
                       'null_class_coef': 0.5,
                       'learning_rate': 1e-3,
                       'batch_size': 2,
                       'n_workers':6})
