"""
====================
Custom Task Tutorial
====================

Computer vision is an umbrella term for a wide spectrum of objectives models are trained for. These objective reflect
on the structure of the data and the possible actions on it.

The first step before running any Deepchecks checks is to create an implementation of
:class:`VisionData <vision_data.VisionData>`. Each implementation represents and standardize a computer vision task
and allows to run a more complex checks which relates to the given task's characteristics. There are default
base classes for a few known tasks like object detection and classification, however not all tasks have a base
implementation, meaning you will have to create your own task.

When creating your own task you will be limited to run checks which are agnostic to the specific task type.
For example performance checks that uses IOU works only on object detection tasks, since they need to know
the exact bounding box format in order to run, while other checks that uses
:doc:`/user-guide/vision/vision_properties` or custom metrics are agnostic to the task type.

In this guide we will implement a custom instance segmentation task and run checks on it.

1. `Defining the Data <#defining-the-data>`__
2. `Implement Custom Task <#implement-custom-task>`__
3. `Implement Custom Properties <#implement-custom-properties>`__
4. `Implement Custom Metric <#implement-custom-metric>`__
"""

#%%
# Defining the Data
# =================
# First we will define a `PyTorch Dataset <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_.
# of COCO-128 segmentation task. This part represents your own code, and is not yet Deepchecks related.

import contextlib
import os
import typing as t
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.utils import draw_segmentation_masks


class CocoSegmentDataset(VisionDataset):
    """An instance of PyTorch VisionData the represents the COCO128-segments dataset.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    name : str
        Name of the dataset.
    train : bool
        if `True` train dataset, otherwise test dataset
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    """

    TRAIN_FRACTION = 0.5

    def __init__(
            self,
            root: str,
            name: str,
            train: bool = True,
            transforms: t.Optional[t.Callable] = None,
    ) -> None:
        super().__init__(root, transforms=transforms)

        self.train = train
        self.root = Path(root).absolute()
        self.images_dir = Path(root) / 'images' / name
        self.labels_dir = Path(root) / 'labels' / name

        images: t.List[Path] = sorted(self.images_dir.glob('./*.jpg'))
        labels: t.List[t.Optional[Path]] = []

        for image in images:
            label = self.labels_dir / f'{image.stem}.txt'
            labels.append(label if label.exists() else None)

        assert len(images) != 0, 'Did not find folder with images or it was empty'
        assert not all(l is None for l in labels), 'Did not find folder with labels or it was empty'

        train_len = int(self.TRAIN_FRACTION * len(images))

        if self.train is True:
            self.images = images[0:train_len]
            self.labels = labels[0:train_len]
        else:
            self.images = images[train_len:]
            self.labels = labels[train_len:]

    def __getitem__(self, idx: int) -> t.Tuple[Image.Image, np.ndarray]:
        """Get the image and label at the given index."""
        image = Image.open(str(self.images[idx]))
        label_file = self.labels[idx]

        masks = []
        classes = []
        if label_file is not None:
            for label_str in label_file.open('r').read().strip().splitlines():
                label = np.array(label_str.split(), dtype=np.float32)
                class_id = int(label[0])
                # Transform normalized coordinates to un-normalized
                coordinates = (label[1:].reshape(-1, 2) * np.array([image.width, image.height])).reshape(-1).tolist()
                # Create mask image
                mask = Image.new('L', (image.width, image.height), 0)
                ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=1)
                # Add to list
                masks.append(np.array(mask, dtype=bool))
                classes.append(class_id)

        if self.transforms is not None:
            # Albumentations accepts images as numpy
            transformed = self.transforms(image=np.array(image), masks=masks)
            image = transformed['image']
            masks = transformed['masks']
            # Transform masks to tensor of (num_masks, H, W)
            if masks:
                masks = torch.stack([torch.from_numpy(m) for m in masks])
            else:
                masks = torch.empty((0, 3))

        return image, classes, masks

    def __len__(self):
        return len(self.images)

    @classmethod
    def load_or_download(cls, root: Path, train: bool) -> 'CocoSegmentDataset':
        coco_dir = root / 'coco128'
        folder = 'train2017'

        if not coco_dir.exists():
            url = 'https://ultralytics.com/assets/coco128-segments.zip'
            md5 = 'e29ec06014d1e06b58b6ffe651c0b34f'

            with open(os.devnull, 'w', encoding='utf8') as f, contextlib.redirect_stdout(f):
                download_and_extract_archive(
                    url,
                    download_root=str(root),
                    extract_root=str(root),
                    md5=md5
                )
            
            try:
                # remove coco128's README.txt so that it does not come in docs
                os.remove("coco128/README.txt")
            except:
                pass
        return CocoSegmentDataset(coco_dir, folder, train=train, transforms=A.Compose([ToTensorV2()]))


# Download and load the datasets
curr_dir = Path('.')
train_ds = CocoSegmentDataset.load_or_download(curr_dir, train=True)
test_ds = CocoSegmentDataset.load_or_download(curr_dir, train=False)


def batch_collate(batch):
    """Function which gets list of samples from `CocoSegmentDataset` and combine them to a batch."""
    images, classes, masks = zip(*batch)
    return list(images), list(classes), list(masks)


# Create DataLoaders
train_data_loader = DataLoader(
    dataset=train_ds,
    batch_size=32,
    shuffle=False,
    collate_fn=batch_collate
)

test_data_loader = DataLoader(
    dataset=test_ds,
    batch_size=32,
    shuffle=False,
    collate_fn=batch_collate
)

#%%
# Visualizing that we loaded our datasets correctly:

masked_images = [draw_segmentation_masks(train_ds[i][0], masks=train_ds[i][2], alpha=0.7)
                 for i in range(5)]

fix, axs = plt.subplots(ncols=len(masked_images), figsize=(20, 20))
for i, img in enumerate(masked_images):
    img = img.detach()
    img = F.to_pil_image(img)
    axs[i].imshow(np.asarray(img))
    axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

fix

#%%
# Implement Custom Task
# =====================
# With our data and model ready we can write the task class.
#
# With the built-in base classes, the checks are accessing directly the values which return from the functions
# `infer_on_batch` and `batch_to_labels` and therefore they require a standard format. But with custom task, these
# functions' values are not used directly, so we can just return our own data as is. On the other hand the  functions
# `batch_to_images` and `get_classes` are used so we will need to make sure our data is in the expected format.

from typing import List, Sequence

from deepchecks.vision import VisionData


class MyCustomSegmentationData(VisionData):
    """Class for loading the COCO segmentation dataset."""

    def get_classes(self, batch_labels) -> List[List[int]]:
        """Return per label a list of classes (by id) in it."""
        # The input `batch_labels` is the result of `batch_to_labels` function.
        return batch_labels[0]

    def batch_to_labels(self, batch):
        """Extract from the batch only the labels. No standard format is needed here."""
        _, classes, masks = batch
        return classes, masks

    def infer_on_batch(self, batch, model, device):
        """Infer on a batch of images. No standard format is needed here."""
        predictions = model.to(device)(batch[0])
        return predictions

    def batch_to_images(self, batch) -> Sequence[np.ndarray]:
        """Convert the batch to a list of images as (H, W, C) 3D numpy array per image."""
        return [tensor.numpy().transpose((1, 2, 0)) for tensor in batch[0]]

#%%
# Now we are able to run checks that use only the image data, since it's in the standard Deepchecks format.
# Let's run SingleFeatureContribution check with our task

from deepchecks.vision.checks import SingleFeatureContribution

# Create our task with the `DataLoader`s we defined before.
train_task = MyCustomSegmentationData(train_data_loader)
test_task = MyCustomSegmentationData(test_data_loader)

result = SingleFeatureContribution().run(train_task, test_task)
result

# Now in order to run more check, we'll need to define custom properties or metrics.
#
# Implement Custom Properties
# ===========================
#
# In order to run checks that are using label or prediction properties we'll have to implement
# a custom :doc:`properties </user-guide/vision/vision_properties>`. We'll write label properties and run a label drift
# check.


from itertools import chain

from deepchecks.vision.checks import TrainTestLabelDrift

# The labels object is the result of `batch_to_labels` function we defined earlier. The property should return a flat
# list of values.


def number_of_detections(labels) -> List[int]:
    """Return a list containing the number of detections per sample in batch."""
    classes, all_masks = labels
    return [sample_masks.shape[0] for sample_masks in all_masks]


def classes_in_labels(labels: List[torch.Tensor]) -> List[int]:
    """Return a list containing the classes in batch."""
    classes, all_masks = labels
    # Flatten list of lists into a single list
    return list(chain.from_iterable(classes))


# We will pass this object as parameter to checks that are using label properties
label_properties = [
    {'name': '# Detections per Label', 'method': number_of_detections, 'output_type': 'discrete'},
    {'name': 'Classes in Labels', 'method': classes_in_labels, 'output_type': 'class_id'}
]


result = TrainTestLabelDrift(label_properties=label_properties).run(train_task, test_task)
result

#%%
# Implement Custom Metric
# =======================
#
# Some checks test the model performance and requires a metric in order to run. When using a custom task you will also
# have to create a custom metric in order for those checks to work, since the built-in metrics don't know to handle
# your data structure. The metrics need to conform to the API of
# `pytorch-ignite <https://pytorch.org/ignite/metrics.html>`_.
