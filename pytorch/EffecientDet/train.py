from argparse import ArgumentParser

import pytorch_lightning as pl
import torch


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
        )

        if replace_head:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            head = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            model.roi_heads.box_predictor = head
        else:
            assert num_classes == 91, "replace_head must be true to change num_classes"

        self.model = model
        self.learning_rate = learning_rate

        #print(model)

        #from torchsummary import summary
        #target = {}
        #target["boxes"] = torch.as_tensor([[0.0,1.0,2.0,3.0]], dtype=torch.float32)
        #target["labels"] =  torch.as_tensor([0], dtype=torch.int64)
        #print(summary(self.model, ((1, 512, 512), target)))
        #model.eval()
        #print(summary(self.model,(1, 512, 512)))
        #model.train()

    def forward(self, x):
        self.model.eval()
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
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--pretrained", type=bool, default=False)
        parser.add_argument("--pretrained_backbone", type=bool, default=True)
        parser.add_argument("--trainable_backbone_layers", type=int, default=3)
        parser.add_argument("--replace_head", type=bool, default=True)

        parser.add_argument("--data_dir", type=str, default="/home/markpp/datasets")
        parser.add_argument("--batch_size", type=int, default=8)
        return parser


def run_cli():
    #from pl_bolts.datamodules import VOCDetectionDataModule
    import os, sys
    sys.path.append('../')
    from datasets.doors.datamodule import DoorDataModule


    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FasterRCNN.add_model_specific_args(parser)

    args = parser.parse_args()

    #datamodule = VOCDetectionDataModule.from_argparse_args(args)
    datamodule = DoorDataModule(data_dir='/home/markpp/github/MapGeneralization/data/Public',
                                batch_size=4,
                                image_size=800)

    args.num_classes = datamodule.num_classes

    model = FasterRCNN(**vars(args))
    trainer = pl.Trainer(gpus=1, max_epochs=100, accumulate_grad_batches=4, weights_summary='full')
    trainer.fit(model, datamodule)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir,"model.pt"))
    torch.save(model, os.path.join(output_dir,'final_model.pkl'))

if __name__ == "__main__":
    run_cli()
