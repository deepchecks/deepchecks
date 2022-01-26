import os.path as osp
from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import pathlib

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import transforms

from deepchecks.vision.utils.image_utils import AlbumentationsTransformWrapper, AlbumentationsTransformsWrapper, \
    AlbumentationImageFolder

current_path = pathlib.Path(__file__).parent.resolve()

def get_trained_imagenet_model():
    model = torchvision.models.resnet18(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    return model


def get_imagenet_dataloaders_albumentations() -> Tuple[DataLoader, DataLoader]:
    import albumentations as A
    # TODO this needs to be downloaded from somewhere into a data dir
    # data_url = "https://url-to/imagenet_trainval_50.pb2.tar"
    data_dir = "/Users/nirbenzvi/code/DeepChecks/ImageNet/Subset"
    # download_and_extract_archive(data_url, './', data_dir)
    data_transforms = {
        "train": A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(224, 224),
                A.HorizontalFlip(),
                A.pytorch.ToTensorV2(),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.pytorch.ToTensorV2()
        ]),
        "val": A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.pytorch.ToTensorV2()
        ])
    }
    val_dataset = AlbumentationImageFolder(root=osp.join(data_dir, "val"),
                              transform=data_transforms["val"])
    train_dataset = AlbumentationImageFolder(root=osp.join(data_dir, "train"),
                                transform=data_transforms["train"])
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=1)
    return train_dataloader, val_dataloader

def get_imagenet_dataloaders() -> Tuple[DataLoader, DataLoader]:
    data_url = "placeholder"
    data_dir = "/Users/nirbenzvi/code/DeepChecks/ImageNet/Subset"
    # download_and_extract_archive(data_url, './', data_dir)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    val_dataset = ImageFolder(root=osp.join(data_dir, "val"), transform=data_transforms["val"])
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    train_dataset = ImageFolder(root=osp.join(data_dir, "train"), transform=data_transforms["train"])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4)
    return train_dataloader, val_dataloader
