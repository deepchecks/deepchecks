# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module for loading a sample of the COCO dataset and the yolov5s model."""
try:
    import tensorflow as tf  # noqa: F401
    import tensorflow_hub as hub  # noqa: F401
except ImportError as error:
    raise ImportError('tensorflow or tensorflow_hub is not installed. Please install them '
                      'in order to use tensorflow coco dataset.') from error

import os
import typing as t
from pathlib import Path

import albumentations as A
import numpy as np
from typing_extensions import Literal

from deepchecks import vision
from deepchecks.vision.datasets.detection.coco_utils import COCO_DIR, LABEL_MAP, download_coco128, get_image_and_label
from deepchecks.vision.vision_data import VisionData

__all__ = ['load_dataset']

_MODEL_URL = 'https://figshare.com/ndownloader/files/38695689'
TRAIN_FRACTION = 0.5
PROBA_THRESHOLD = 0.5


def load_dataset(
        train: bool = True,
        shuffle: bool = False,
        object_type: Literal['VisionData', 'Dataset'] = 'Dataset',
        n_samples: t.Optional[int] = None,
) -> t.Union[tf.data.Dataset, vision.VisionData]:
    """Get the COCO128 dataset and return a dataloader.

    Parameters
    ----------
    train : bool, default: True
        if `True` train dataset, otherwise test dataset
    shuffle : bool, default: False
        Whether to shuffle the dataset.
    object_type : Literal['Dataset', 'Dataset'], default: 'Dataset'
        type of the return value. If 'Dataset', :obj:`deepchecks.vision.VisionData`
        will be returned, otherwise :obj:`tf.data.Dataset`.
    n_samples : int, optional
        Number of samples to load. Return the first n_samples if shuffle
        is False otherwise selects n_samples at random. If None, returns all samples.

    Returns
    -------
    Union[Dataset, VisionData]
        A Dataset or VisionData instance representing COCO128 dataset
    """
    transforms = A.Compose([A.NoOp()], bbox_params=A.BboxParams(format='coco'))
    coco_dataset = create_tf_dataset(train, n_samples, transforms)
    if shuffle:
        coco_dataset = coco_dataset.shuffle(128)

    if object_type == 'Dataset':
        return coco_dataset
    elif object_type == 'VisionData':
        model = hub.load(_MODEL_URL)
        coco_dataset = coco_dataset.map(deepchecks_map(model))
        return VisionData(batch_loader=coco_dataset, label_map=LABEL_MAP, task_type='object_detection',
                          reshuffle_data=False)
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')


def _prediction_to_deepchecks_format(model, image):
    pred = model([image])
    boxes, class_id, prob = pred['detection_boxes'][0], pred['detection_classes'][0], pred['detection_scores'][0]
    # convert [y_min, x_min, y_max, x_max] to [x_min, y_min, w, h]
    formatted_boxes = [boxes[:, 1], boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0], prob, class_id]
    return tf.stack(formatted_boxes, axis=1)[prob > PROBA_THRESHOLD]


def deepchecks_map(model):
    def _deepchecks_map(image, label):
        pred = _prediction_to_deepchecks_format(model, image)
        # class_id is required to be the first column
        label = tf.gather(label, [4, 0, 1, 2, 3], axis=1) if label is not None and len(label) > 0 else label
        return {'images': [image], 'labels': [label], 'predictions': [pred]}

    return _deepchecks_map


def create_tf_dataset(train: bool = True, n_samples: t.Optional[int] = None, transforms=None) -> tf.data.Dataset:
    """Create a tf dataset of the COCO128 dataset."""
    coco_dir, dataset_name = download_coco128(COCO_DIR)
    img_dir = Path(coco_dir / 'images' / dataset_name)
    label_dir = Path(coco_dir / 'labels' / dataset_name)
    files = os.listdir(img_dir)
    train_len = int(TRAIN_FRACTION * len(files))
    files = files[:train_len] if train else files[train_len:]
    if n_samples is not None and n_samples < len(files):
        files = files[:n_samples]

    images, labels = [], []
    for file_name in files:
        label_file = label_dir / str(file_name).replace('jpg', 'txt')
        image, label = get_image_and_label(img_dir / str(file_name), label_file, transforms)
        images.append(image)
        labels.append(np.asarray(label))

    def generator():
        for img, label in zip(images, labels):
            yield img, label

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, None, 3)), tf.TensorSpec(shape=None)))
    return dataset
