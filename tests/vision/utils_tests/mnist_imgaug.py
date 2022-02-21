# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
import numpy as np
import torch
from torchvision import datasets
import typing as t
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader


from deepchecks.vision import VisionData
from deepchecks.vision.datasets.classification.mnist import MODULE_DIR
from deepchecks.vision.utils import ImageFormatter, ClassificationLabelFormatter
from deepchecks.vision.utils.transformations import un_normalize_batch


class MNIST(datasets.MNIST):
    """MNIST Dataset."""

    def __getitem__(self, index: int) -> t.Tuple[t.Any, t.Any]:
        """Get sample."""
        # NOTE:
        # we use imgaug for an image augmentation
        # which requires the image to be passed to the transform function as
        # an numpy array, because of this we overridded this method

        img, target = self.data[index].numpy(), int(self.targets[index])

        if self.transform is not None:
            # Imgaug must have channels dimension
            img = np.expand_dims(img, 2)
            img = self.transform(image=img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.Tensor(img), target


def mnist_dataset_imgaug(train: bool = True, dataset=None):
    mean = (0.1307,)
    std = (0.3081,)

    if dataset is None:
        dataset = MNIST(
            str(MODULE_DIR),
            train=train,
            download=True,
            transform=iaa.Sequential([
                iaa.Lambda(func_images=normalize(mean, std))
            ]),
        )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    return VisionData(
        data_loader=loader,
        num_classes=len(loader.dataset.classes),
        label_formatter=ClassificationLabelFormatter(),
        image_formatter=ImageFormatter(lambda batch: un_normalize_batch(batch[0], mean, std)),
        transform_field='transform'
    )


def normalize(mean, std):
    # pylint: disable=unused-argument
    def func(images, random_state, parents, hooks):
        max_pixel_value = np.array(255)
        return [(img / max_pixel_value - np.array(mean)) / np.array(std) for img in images]
    return func
