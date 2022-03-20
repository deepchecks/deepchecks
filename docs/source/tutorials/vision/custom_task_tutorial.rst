====================
Custom Task Tutorial
====================

Computer vision is consists of different objectives models are trained for. The desired objective reflects
on the structure of the data and the possible actions on it.

In Deepchecks the first step before running checks is to create an implementation of `VisionData` which
standardize the structure of the data for a given computer vision task.

Some of the checks are agnostic to the specific task and will work on all of them, and some are tied to the
specific details of a task and won't work for other tasks. When implementing your own task all of the
task-agnostic checks will be able to run.

Defining the Data & Model
=========================

First we will define a `Dataset` of COCO-128 segmentation task.

.. code-block:: python

  import os
  import contextlib
  import typing as t
  from pathlib import Path

  from torchvision.datasets import VisionDataset
  from torchvision.datasets.utils import download_and_extract_archive
  from PIL import Image, ImageDraw
  import numpy as np
  import albumentations as A
  from albumentations.pytorch.transforms import ToTensorV2


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
          img = Image.open(str(self.images[idx]))
          label_file = self.labels[idx]

          masks = []
          classes = []
          if label_file is not None:
              for label_str in label_file.open('r').read().strip().splitlines():
                  label = np.array(label_str.split(), dtype=np.float32)
                  class_id = label[0]
                  # Transform normalized coordinates to un-normalized
                  coordinates = (label[1:].reshape(-1, 2) * np.array([img.width, img.height])).reshape(-1).tolist()
                  # Create mask image
                  mask = Image.new('L', (img.width, img.height), 0)
                  ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=1)
                  # Add to list
                  masks.append(np.array(mask, dtype=bool))
                  classes.append(class_id)

          if self.transforms is not None:
              # Albumentations accepts images as numpy
              transformed = self.transforms(image=np.array(img), masks=masks)
              img = transformed['image']
              masks = transformed['masks']
              # Transform masks to tensor of (num_masks, H, W)
              masks = torch.stack([torch.from_numpy(m) for m in masks])

          return img, classes, masks

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
          return CocoSegmentDataset(coco_dir, folder, train=train, transforms=A.Compose([ToTensorV2()]))


Now we'll download and create the `Dataset` and `DataLoader` of both train and test:

.. code-block:: python

  from torch.utils.data import DataLoader
  from pathlib import Path

  root = Path('.')
  train_ds = CocoSegmentDataset.get_download(root, train=True)
  test_ds = CocoSegmentDataset.get_download(root, train=False)

  def batch_collate(batch):
      imgs, classes, masks = zip(*batch)
      return list(imgs), list(classes), list(masks)

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


Visualizing that our dataset was loaded correctly:

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt

  from torchvision.utils import draw_segmentation_masks
  import torchvision.transforms.functional as F

  def show(imgs):
      if not isinstance(imgs, list):
          imgs = [imgs]
      fix, axs = plt.subplots(ncols=len(imgs), figsize=(20,20))
      for i, img in enumerate(imgs):
          img = img.detach()
          img = F.to_pil_image(img)
          axs[i].imshow(np.asarray(img))
          axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


  imgs = [draw_segmentation_masks(train_ds[i][0], masks=train_ds[i][2], alpha=0.8)
          for i in range(5)]

  show(imgs)

.. image :: /_static/custom-segment-tutorial.png
  :width: 400
  :alt: COCO128 segmentation visualize

For the model we will use a pretrain model from `torchvision`

.. code-block:: python

  from torchvision.models.detection import fasterrcnn_resnet50_fpn

  model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
  model = model.eval()


Implement Custom `VisionData`
=============================

When we have our `DataLoader`s and model ready we can start creating our own implementation
of `VisionData` class.

.. code-block:: python

  from typing import List, Iterable
  from deepchecks.vision.vision_data import VisionData

  class MyCustomSegmentationData(VisionData):
      """Class for loading the COCO segmentation dataset."""

      def get_classes(self, batch_labels) -> List[List[int]]:
          """Return per label a list of classes (by id) in it."""
          # The input `batch_labels` is the result of `batch_to_labels` function.
          return batch_labels[0]

      def batch_to_labels(self, batch):
          """Extract from the batch only the labels."""
          return batch[1], batch[2]

      def infer_on_batch(self, batch, model, device):
          """Infer on a batch of images."""
          predictions = model.to(device)(batch[0])
          return predictions

      def batch_to_images(self, batch) -> Iterable[np.ndarray]:
          """Convert the batch to a list of images as (H, W, C) 3D numpy array per image."""
          return [tensor.numpy().transpose((1, 2, 0)) for tensor in batch[0]]

Now we are able to run checks that are working only on the image data, since it's in the standard format of
Deepchecks. For more checks we'll need to define custom properties or metrics.

Implement Custom Properties
===========================

In order to run checks that are using label or prediction properties we we'll have to implement
a custom property. For a more in-depth explanation on properties see
:doc:`Data Properties </user-guide/vision/vision_properties>`

.. code-block:: python

  #### Label properties

  def number_of_detections(labels) -> List[int]:
  """Return a list containing the number of detections per sample in batch."""
  # Labels object is the result of `batch_to_labels` function
  classes, all_masks = labels
  return [sample_masks.shape[0] for sample_masks in all_masks]

  def classes_in_labels(labels: List[torch.Tensor]) -> List[int]:
    """Return a list containing the classes in batch."""
    # Labels object is the result of `batch_to_labels` function
    classes, all_masks = labels
    return list(chain.from_iterable(classes))

  # We will pass this object as parameter to checks that are using label properties
  label_properties = [
    {'name': '# Detections per Label', 'method': number_of_detections, 'output_type': 'discrete'},
    {'name': 'Classes in Labels', 'method': classes_in_labels, 'output_type': 'class_id'}
  ]


Implement Custom Metric
=======================

Some checks requires a metric in order to run. When using a custom task you will also have to create a custom
metric in order for those checks to work, since the built-in metrics does not know to handle your data structure.
The metrics need to conform to the API of
`pytorch-ignite <https://pytorch.org/ignite/metrics.html>`_.

