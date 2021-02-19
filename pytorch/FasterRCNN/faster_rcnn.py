from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import numpy as np
import cv2

try:
    import torchvision.utils as vutils
    from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn
    from torchvision.ops import box_iou
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()

def _plot_boxes(imgs, targets, preds):
    """
    Plot the target and prediction boxes
    """
    dets = []
    for img, tar, pred in zip(imgs, targets, preds):
        out = img.cpu().mul(255).permute(1, 2, 0).byte().numpy()
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

class FasterRCNN(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Faster R-CNN: Towards Real-Time Object Detection with
    Region Proposal Networks <https://arxiv.org/abs/1506.01497>`_.

    Paper authors: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun

    Model implemented by:
        - `Teddy Koker <https://github.com/teddykoker>`

    During training, the model expects both the input tensors, as well as targets (list of dictionary), containing:
        - boxes (`FloatTensor[N, 4]`): the ground truth boxes in `[x1, y1, x2, y2]` format.
        - labels (`Int64Tensor[N]`): the class label for each ground truh box

    CLI command::

        # PascalVOC
        python faster_rcnn.py --gpus 1 --pretrained True
    """
    def __init__(
        self,
        learning_rate: float = 0.0001,
        num_classes: int = 91,
        pretrained: bool = False,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        replace_head: bool = True,
        **kwargs,
    ):
        """
        Args:
            learning_rate: the learning rate
            num_classes: number of detection classes (including background)
            pretrained: if true, returns a model pre-trained on COCO train2017
            pretrained_backbone: if true, returns a model with backbone pre-trained on Imagenet
            trainable_backbone_layers: number of trainable resnet layers starting from final block
        """
        super().__init__()

        model = fasterrcnn_resnet50_fpn(
            # num_classes=num_classes,
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
            image_mean = [0.5,0.5,0.5],
            image_std = [0.5,0.5,0.5]
        )

        if replace_head:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            head = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            model.roi_heads.box_predictor = head
        else:
            assert num_classes == 91, "replace_head must be true to change num_classes"

        self.model = model
        self.learning_rate = learning_rate

        #self.log_hyperparams(hparms)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)

        if batch_idx == 0:
            dets = _plot_boxes(images, targets, outs)
            self.logger.experiment.add_image("images", dets, self.current_epoch, dataformats='HWC')

        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        self.log('val_iou', avg_iou, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.005,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0002)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--pretrained", type=bool, default=False)
        parser.add_argument("--pretrained_backbone", type=bool, default=False)
        parser.add_argument("--trainable_backbone_layers", type=int, default=5)
        parser.add_argument("--replace_head", type=bool, default=True)

        parser.add_argument("--data_dir", type=str, default="/home/markpp/datasets")
        parser.add_argument("--batch_size", type=int, default=8)
        return parser


def run_cli():

    import os, sys
    sys.path.append('../')

    #'''
    from datasets.doors.datamodule import DoorDataModule
    datamodule = DoorDataModule(data_dir='/home/markpp/github/MapGeneralization/data/Public',
                                batch_size=4,
                                image_size=800)
    #'''

    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FasterRCNN.add_model_specific_args(parser)

    args = parser.parse_args()

    '''
    from datasets.VOC.vocdetection_datamodule import VOCDetectionDataModule
    datamodule = VOCDetectionDataModule.from_argparse_args(args)
    '''
    args.num_classes = datamodule.num_classes

    model = FasterRCNN(**vars(args))

    trainer = pl.Trainer(gpus=1, max_epochs=1000, accumulate_grad_batches=4)#, weights_summary='full') #val_percent_check=0.1,
    trainer.fit(model, datamodule)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.model.state_dict(), os.path.join(output_dir,"model.pt"))
    torch.save(model.model, os.path.join(output_dir,'final_model.pkl'))

if __name__ == "__main__":
    run_cli()
