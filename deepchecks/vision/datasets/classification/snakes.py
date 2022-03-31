import os.path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local
from torch import nn
from torchvision.datasets.utils import download_and_extract_archive

from deepchecks.vision.nir_mixed.snake_lit_module import SnakeLitModule
from deepchecks.vision.utils.image_utils import AlbumentationImageFolder

DATASET_ROOT = os.path.join(os.path.dirnam(__file__), 'snakes')
SNAKE_CKPT = os.path.join(DATASET_ROOT, 'snakes.ckpt')


def load_val_data(transformed=False):
    data_dir = os.path.join(DATASET_ROOT, "test")
    dataset = AlbumentationImageFolder(root=data_dir)
    if transformed:
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
        dataset.transforms = transforms
    return dataset


def load_train_data(transformed=False):
    data_dir = os.path.join(DATASET_ROOT, "train")
    dataset = AlbumentationImageFolder(root=data_dir)
    if transformed:
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.RandomCrop(height=224, width=224),
            A.HorizontalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
        dataset.transforms = transforms
    return dataset


def load_model(pretrained: bool = True, device: str = "cpu") -> nn.Module:
    model = SnakeLitModule(optimizer="adam",
                           finetune_last=True,
                           )
    if pretrained:
        model = model.load_from_checkpoint(checkpoint_path=SNAKE_CKPT)
    model = model.eval()
    model = model.to(device)
    return model


def _download_snake_data(snake_data_url: str, snake_train_list: str, snake_val_list: str, data_dir: str):
    # TODO implement this; get dataset from Kaggle and use supplied lists to split it online
    # download_and_extract_archive(snake_data_url, './', data_dir)
    pass
