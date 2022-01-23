"""Module for loading a sample of the COOC dataset and the yolov5s model."""
import glob
import os
from pathlib import Path
from typing import List, Any, Tuple, Optional, Callable

import numpy as np
import torch
from PIL import Image
from numpy import genfromtxt
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


def get_trained_yolov5_object_detection():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    model.cpu()

    return model


class CocoDataset(VisionDataset):

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super(CocoDataset, self).__init__(root, transforms, transform, target_transform)
        images_dir = Path(root) / 'images' / name
        labels_dir = Path(root) / 'labels' / name
        self.images = [n for n in images_dir.iterdir()]
        self.labels = []
        for image in self.images:
            base, _ = os.path.splitext(os.path.basename(image))
            label = labels_dir / f'{base}.txt'
            self.labels.append(label if label.exists() else None)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[idx]).convert('RGB')

        label_file = self.labels[idx]
        if label_file is not None:  # found
            with open(label_file, 'r') as f:
                labels = [x.split() for x in f.read().strip().splitlines()]
                labels = np.array(labels, dtype=np.float32)
        else:  # missing
            labels = np.zeros((0, 5), dtype=np.float32)

        boxes = []
        classes = []
        for i in range(len(labels)):
            x, y, w, h = labels[i, 1:]
            labels[i, 1:] = np.array([
                (x - w / 2) * img.width,
                (y - h / 2) * img.height,
                w * img.width,
                h * img.height])

        if self.transforms is not None:
            img, labels = self.transforms(img, labels)

        return img, labels

    def __len__(self) -> int:
        return len(self.images)


def get_coco_dataloader():
    data_url = 'https://ultralytics.com/assets/coco128.zip'
    download_and_extract_archive(data_url, './', './coco128')
    dataset = CocoDataset(os.path.join('coco128', 'coco128'), 'train2017')

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                            collate_fn=lambda batch: tuple(zip(*batch)))

    return dataloader
