from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from model import RetinaNetModel

#https://github.com/benihime91/pytorch_retinanet

# load in the hparams yaml file
hparams = OmegaConf.load("hparams.yaml")

# instantiate lightning module
model = RetinaNetModel(hparams)

# Instantiate Trainer
trainer = Trainer(gpus=1, accumulate_grad_batches=8)
# start train
trainer.fit(model)
# to test model using COCO API
trainer.test(model)
