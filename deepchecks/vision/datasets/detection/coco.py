"""Module for loading a sample of the COOC dataset and the yolov5s model."""
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import functional as func
from torchvision.datasets import VisionDataset
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

    def _load_image(self, idx: int) -> Image.Image:
        return func.to_tensor(Image.open(self.files[idx]).convert("RGB")).to('cpu')

    def _load_target(self, idx: int) -> List[Any]:
        return genfromtxt(self.labels[idx])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        idx = self.ids[index]
        image = self._load_image(idx)
        target = self._load_target(idx)

        return image, target

    def __len__(self):
        return len(self.ids)


def get_coco_dataloader():
    data_url = 'https://ultralytics.com/assets/coco128.zip'
    download_and_extract_archive(data_url, './', './coco128')
    return DataLoader(CocoDataset('./coco128/'), batch_size=1)

