"""Module for loading a sample of the COOC dataset and the yolov5s model."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms
import glob
import os
from numpy import genfromtxt
from PIL import Image
from typing import List, Any, Tuple


def get_trained_yolov5_object_detection():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    model.cpu()

    return model


class CocoDataset(VisionDataset):

    def __init__(self, path):
        super().__init__(path)
        self.file_path = os.path.join(path, 'coco128/images/train2017')
        self.label_path = os.path.join(path, 'coco128/labels/train2017')
        self.files = sorted(glob.glob(os.path.join(self.file_path, '*.jpg')))
        self.labels = sorted(glob.glob(os.path.join(self.label_path, '*.txt')))
        self.ids = list(range(len(self.files)))

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((480, 640)),
            transforms.ToPILImage()
        ])

    def _load_image(self, idx: int) -> Image.Image:
        return Image.open(self.files[idx]).convert("RGB")

    def _load_target(self, idx: int) -> List[Any]:
        return genfromtxt(self.labels[idx])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        idx = self.ids[index]
        image = self._load_image(idx)
        target = self._load_target(idx)

        return torch.tensor(np.array(self._transform(image))).to('cpu'), target.reshape(-1, 5)

    def __len__(self):
        return len(self.ids)


def get_coco_dataloader():
    data_url = 'https://ultralytics.com/assets/coco128.zip'
    download_and_extract_archive(data_url, './', './coco128')
    return DataLoader(CocoDataset('./coco128/'), batch_size=1)

