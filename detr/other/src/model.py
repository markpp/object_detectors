import torch
import torch.nn as nn
# I wrote a function to get the DETR Model from torch hub
# I am planning to add support for pytorch image models by ross as backbones
# I load the Pytorch hub model from detr

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

__all__ = ["detr_model"]
class detr_model(nn.Module):
    def __init__(self, n_classes, n_queries, backbone, pretrained=True):
        super().__init__()
        self.n_queries = n_queries
        self.n_classes = n_classes

        self.model = torch.hub.load('facebookresearch/detr', backbone, pretrained=True)
        self.in_features =  self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.n_classes)
        self.model.n_classes = self.n_classes
        self.model.n_queries = self.n_queries

    def forward(self, images):
        return self.model(images)
