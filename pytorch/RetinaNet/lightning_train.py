
 import pytorch_lightning as pl
 from pytorch_lightning import Trainer

 from lightning_model import RetinaNetModel

 # load in the hparams yaml file
 hparams = OmegaConf.load("hparams.yaml")

 # instantiate lightning module
 model = RetinaNetModel(hparams=hparams)

 # Instantiate Trainer
 trainer = Trainer()
 # start train
 trainer.fit(model)
 # to test model using COCO API
 trainer.test(model)
