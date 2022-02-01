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
"""Module representing the MNIST dataset."""
import typing as t
import pathlib
import logging

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch import nn
from torch.utils.data import DataLoader
from typing_extensions import Literal

from deepchecks.vision.dataset import VisionDataset
from . import MODELS_DIR


__all__ = ['load_dataset', 'load_model', 'MNistNet', 'MNIST']


LOGGER = logging.getLogger(__name__)


MODULE_DIR = pathlib.Path(__file__).absolute().parent
DATA_PATH = MODULE_DIR / 'MNIST'
MODEL_PATH = MODELS_DIR / 'mnist.pth'


def load_dataset(
    train: bool = True,
    batch_size: t.Optional[int] = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    object_type: Literal['Dataset', 'DataLoader'] = 'DataLoader'
) -> t.Union[DataLoader, VisionDataset]:
    """Download MNIST dataset.

    Parameters
    ----------
    train : bool, default True
        Train or Test dataset
    batch_size: int, optional
        how many samples per batch to load
    shuffle : bool, default ``True``
        to reshuffled data at every epoch or not
    pin_memory : bool, default ``True``
        If ``True``, the data loader will copy Tensors
        into CUDA pinned memory before returning them.
    object_type : Literal[Dataset, DataLoader], default 'DataLoader'
        object type to return. if `'Dataset'` then :obj:`deepchecks.vision.VisionDataset`
        will be returned, if `'DataLoader'` then :obj:`torch.utils.data.DataLoader`

    Returns
    -------
    Union[:obj:`deepchecks.vision.VisionDataset`, :obj:`torch.utils.data.DataLoader`]

        depending on the ``object_type`` parameter value, instance of
        :obj:`deepchecks.vision.VisionDataset` or :obj:`torch.utils.data.DataLoader`
        will be returned

    """
    batch_size = batch_size or (64 if train else 1000)

    loader = DataLoader(
        MNIST(
            str(MODULE_DIR),
            train=train,
            download=True,
            transform=A.Compose([
                # TODO: what else transformations we need to apply
                A.Normalize((0.1307,), (0.3081,)),
                ToTensorV2(),
            ]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory
    )

    if object_type == 'DataLoader':
        return loader
    elif object_type == 'Dataset':
        return VisionDataset(
            data_loader=loader,
            num_classes=len(datasets.MNIST.classes)
        )
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')


def load_model(pretrained: bool = True) -> 'MNistNet':
    """Load MNIST model.

    Returns
    -------
    MNistNet
    """
    # TODO: should we put downloadable pre-trained model into our repo?

    if pretrained and MODEL_PATH.exists():
        model = MNistNet()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        return model

    model = MNistNet()

    if pretrained is False:
        return model.train()

    dataloader = t.cast(DataLoader, load_dataset(train=True, object_type='DataLoader'))
    datasize = len(dataloader.dataset)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 5

    model.train()

    LOGGER.info('== Starting model training ==')

    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):  # pylint: disable=invalid-name
            X, y = X.to('cpu'), y.to('cpu')  # pylint: disable=invalid-name

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                LOGGER.info(
                    'Epoch: %f; loss=%f, %d/%d',
                    epoch, loss, current, datasize
                )

    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir()

    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    return model


class MNIST(datasets.MNIST):
    """MNIST Dataset."""

    def __getitem__(self, index: int) -> t.Tuple[t.Any, t.Any]:
        """Get sample."""
        # NOTE:
        # we use albumentations for an image augmentation
        # which requires the image to be passed to the transform function as
        # an numpy array, because of this we overridded this method

        img, target = self.data[index].numpy(), int(self.targets[index])

        if self.transform is not None:
            img = self.transform(image=img)['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNistNet(nn.Module):
    """Represent a simple MNIST network."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """Run a forward step on the network."""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
