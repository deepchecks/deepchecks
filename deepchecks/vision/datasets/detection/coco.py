"""Module for loading a sample of the COCO dataset and the yolov5s model."""
import os
from pathlib import Path
from typing import List, Any, Tuple, Optional, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


def get_trained_yolov5_object_detection():
    """Load the yolov5s model and return it."""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    model.cpu()

    return model


class CocoDataset(VisionDataset):
    """An instance of PyTorch VisionDataset the represents the COCO dataset.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    name : str
        Name of the dataset.
    transform : Callable, optional
        A function/transforms that takes in an image and a label and returns the
        transformed versions of both.
        E.g, ``transforms.Rotate``
    target_transform : Callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        images_dir = Path(root) / 'images' / name
        labels_dir = Path(root) / 'labels' / name
        self.images = [n for n in images_dir.iterdir() if n.name.endswith('.jpg')]
        self.labels = []
        for image in self.images:
            base, _ = os.path.splitext(os.path.basename(image))
            label = labels_dir / f'{base}.txt'
            self.labels.append(label if label.exists() else None)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get the image and label at the given index."""
        img = Image.open(self.images[idx]).convert('RGB')

        label_file = self.labels[idx]
        if label_file is not None:  # found
            with open(label_file, 'r', encoding='utf8') as f:
                labels = [x.split() for x in f.read().strip().splitlines()]
                labels = np.array(labels, dtype=np.float32)
        else:  # missing
            labels = np.zeros((0, 5), dtype=np.float32)

        # Transform x,y,w,h in yolo format (x, y are of the image center, and coordinates are normalized) to standard
        # x,y,w,h format, where x,y are of the top left corner of the bounding box and coordinates are absolute.
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
        """Return the number of images in the dataset."""
        return len(self.images)


def get_coco_dataloader(batch_size: int = 64, num_workers: int = 0, shuffle: bool = False) -> DataLoader:
    """Get the COCO dataset and return a dataloader.

    Parameters
    ----------
    batch_size : int, default: 64
        Batch size for the dataloader.
    num_workers : int, default: 0
        Number of workers for the dataloader.
    shuffle : bool, default: False
        Whether to shuffle the dataset.

    Returns
    -------
    DataLoader
        A dataloader for the COCO dataset.
    """
    if not os.path.exists(os.path.join(os.getcwd(), 'coco128', 'coco128')):
        data_url = 'https://ultralytics.com/assets/coco128.zip'
        download_and_extract_archive(data_url, './', './coco128')
    dataset = CocoDataset(os.path.join('coco128', 'coco128'), 'train2017')

    def batch_collate(batch):
        imgs, labels = zip(*batch)
        return list(imgs), list(labels)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, collate_fn=batch_collate)

    return dataloader


def yolo_wrapper(predictions: 'ultralytics.models.common.Detections') -> List[torch.Tensor]:  # noqa: F821
    """Convert from yolo Detections object to List (per image) of Tensors of the shape [N, 6] with each row being \
    [x, y, w, h, confidence, class] for each bbox in the image."""
    return_list = []
    for single_image_tensor in predictions.pred:  # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
        pred_modified = torch.clone(single_image_tensor)
        pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]  # w = x_right - x_left
        pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]  # h = y_bottom - y_top

        return_list.append(pred_modified)
    return return_list
