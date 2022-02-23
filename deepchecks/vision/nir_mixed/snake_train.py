import datetime
import os
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A
import sys

# Ugly hack
sys.path.insert(0, os.getcwd())
# local
from deepchecks.vision.nir_mixed.snake_skewed_data_module import SnakeSkewedDataModule
from deepchecks.vision.nir_mixed.snake_lit_module import SnakeLitModule

# Train parameters, # TODO these should be CLIs
num_gpus = torch.cuda.device_count()
batch_size = 256
num_workers = 8
torch.manual_seed(42)

# Create tensorboard logger
logger = TensorBoardLogger("workdir", version=1, name=None)
# This takes care of checkpointing in Lightning
checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir,
                                      every_n_train_steps=None,
                                      save_last=True,
                                      save_top_k=-1,
                                      every_n_epochs=1)
# Create train transform
train_transforms = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.RandomCrop(height=224, width=224),
    A.HorizontalFlip(),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2()
])
# validation has center crop, train has random crop + flip, otherwise same
val_transforms = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])
# Choose this for splitting live (split is saved after training ends)
snake_module = SnakeSkewedDataModule(data_dir=os.path.expanduser("~/code/DeepChecks/Datasets/snakes/original"),
                                     batch_size=batch_size,
                                     train_transforms=train_transforms,
                                     val_transforms=val_transforms,
                                     num_workers=num_workers,
                                     subset_size=5000,
                                     skew_class=0,
                                     skew_ratio=0.3)
# Choose this for pre-split datasets
# snake_module = SnakeDataModule(train_data_dir=os.path.expanduser("~/code/DeepChecks/Datasets/snakes/train"),
#                                val_data_dir=os.path.expanduser("~/code/DeepChecks/Datasets/snakes/val"),
#                                batch_size=batch_size,
#                                train_transforms=train_transforms,
#                                val_transforms=val_transforms,
#                                num_workers=num_workers)
# Define LightningModule for training
net = SnakeLitModule(num_classes=snake_module.num_classes,
                     optimizer="adam",
                     finetune_last=True,
                     )

# Trainer Object
trainer = Trainer(logger=logger, gpus=num_gpus, callbacks=[checkpoint_callback], enable_checkpointing=True,
                  max_epochs=200)
# Model.fit
trainer.fit(net, datamodule=snake_module)
# This saves the split into the logging folder for future use
snake_module.save_data_partitions(logger.log_dir)
