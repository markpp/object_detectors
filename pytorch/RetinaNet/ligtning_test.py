
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from model import RetinaNetModel

# load in the hparams yaml file
hparams =

# create datamodule

# instantiate lightning module
model = RetinaNetModel(hparams=hparams)

# Instantiate Trainer
trainer = Trainer()
# start train
trainer.fit(model)
#
#trainer.test(model)
