import os
import shutil

import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A
import sys

# local
# Ugly hack
sys.path.insert(0, os.getcwd())
from deepchecks.vision.nir_mixed.snake_data_module import SnakeDataModule
from deepchecks.vision.nir_mixed.snake_lit_module import SnakeLitModule

# Train parameters, # TODO these should be CLIs
num_gpus = torch.cuda.device_count()
batch_size = 256
num_workers = 8
torch.manual_seed(42)

logger = TensorBoardLogger("workdir", name=None)
checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir,
                                      monitor="val_acc",
                                      save_last=True,
                                      save_top_k=True,
                                      every_n_epochs=5)
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
snake_module = SnakeDataModule(data_dir=os.path.expanduser("~/code/DeepChecks/Datasets/snakes/original"),
                               batch_size=batch_size,
                               train_transforms=train_transforms,
                               val_transforms=val_transforms,
                               num_workers=num_workers)
snake_module.setup()
# Now save
train_dir = os.path.join(snake_module.data_dir, "..", "train")
val_dir = os.path.join(snake_module.data_dir, "..", "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def save_subset(subset, dataset_dir):
    data_list = []
    for subset_idx in subset.indices:
        sample = snake_module.dataset.samples[subset_idx]
        img_path, label = sample
        target_dir = os.path.join(dataset_dir, f"class-{str(label)}")
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(img_path, target_dir)
        data_list.append("\t".join(list(map(str, sample))) + "\n")
    with open(f"{dataset_dir}.txt", "w") as fid:
        [fid.write(row_str) for row_str in data_list]

save_subset(snake_module.val, val_dir)
save_subset(snake_module.train, train_dir)


