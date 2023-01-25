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
"""Module representing the MNIST dataset in pytorch."""
try:
    from torchvision import datasets
    from torchvision.datasets.mnist import read_image_file, read_label_file
    from torchvision.datasets.utils import check_integrity, download_and_extract_archive
except ImportError as error:
    raise ImportError('torchvision is not installed. Please install torchvision>=0.11.3 '
                      'in order to use the selected dataset.') from error
import logging
import os
import pathlib
import pickle
import typing as t
import warnings
from itertools import cycle
from urllib.error import URLError

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from typing_extensions import Literal

from deepchecks.utils.logger import get_logger
from deepchecks.vision.utils.test_utils import get_data_loader_sequential, hash_image, un_normalize_batch
from deepchecks.vision.vision_data import BatchOutputFormat, VisionData
from deepchecks.vision.vision_data.utils import object_to_numpy

__all__ = ['load_dataset', 'load_model', 'MnistModel', 'TorchMnistDataset', 'IterableTorchMnistDataset']

MNIST_DIR = pathlib.Path(__file__).absolute().parent.parent / 'assets' / 'mnist'
MODEL_PATH = MNIST_DIR / 'mnist_model.pth'

LOGGER = logging.getLogger(__name__)


def load_dataset(train: bool = True, batch_size: t.Optional[int] = None, shuffle: bool = False, pin_memory: bool = True,
                 object_type: Literal['VisionData', 'DataLoader'] = 'DataLoader', use_iterable_dataset: bool = False,
                 n_samples=None, device: t.Union[str, torch.device] = 'cpu') -> t.Union[DataLoader, VisionData]:
    """Download MNIST dataset.

    Parameters
    ----------
    train : bool, default : True
        Train or Test dataset
    batch_size: int, optional
        how many samples per batch to load
    shuffle : bool , default : False
        to reshuffled data at every epoch or not, cannot work with use_iterable_dataset=True
    pin_memory : bool, default : True
        If ``True``, the data loader will copy Tensors
        into CUDA pinned memory before returning them.
    object_type : Literal[Dataset, DataLoader], default 'DataLoader'
        object type to return. if `'VisionData'` then :obj:`deepchecks.vision.VisionData`
        will be returned, if `'DataLoader'` then :obj:`torch.utils.data.DataLoader`
    use_iterable_dataset : bool, default False
        if True, will use :obj:`IterableTorchMnistDataset` instead of :obj:`TorchMnistDataset`
    n_samples : int, optional
        Only relevant for loading a VisionData. Number of samples to load. Return the first n_samples if shuffle
        is False otherwise selects n_samples at random. If None, returns all samples.
    device : t.Union[str, torch.device], default : 'cpu'
        device to use in tensor calculations
    Returns
    -------
    Union[:obj:`deepchecks.vision.VisionData`, :obj:`torch.utils.data.DataLoader`]

        depending on the ``object_type`` parameter value, instance of
        :obj:`deepchecks.vision.VisionData` or :obj:`torch.utils.data.DataLoader`
        will be returned

    """
    batch_size = batch_size or (64 if train else 1000)
    transform = A.Compose([A.Normalize(mean=(0.1307,), std=(0.3081,)), ToTensorV2()])
    if use_iterable_dataset:
        dataset = IterableTorchMnistDataset(train=train, transform=transform, n_samples=n_samples)
    else:
        dataset = TorchMnistDataset(str(MNIST_DIR), train=train, download=True, transform=transform)

    if object_type == 'DataLoader':
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                          generator=torch.Generator())
    elif object_type == 'VisionData':
        model = load_model(device=device)
        loader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, generator=torch.Generator(),
                            collate_fn=deepchecks_collate(model))
        if not use_iterable_dataset:
            loader = get_data_loader_sequential(loader, shuffle, n_samples)
        return VisionData(loader, task_type='classification', reshuffle_data=False)
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')


def collate_without_model(data) -> t.Tuple[t.List[np.ndarray], t.List[int]]:
    """Collate function for the mnist dataset returning images and labels in correct format as tuple."""
    raw_images = torch.stack([x[0] for x in data])
    labels = [x[1] for x in data]
    images = raw_images.permute(0, 2, 3, 1)
    images = un_normalize_batch(images, mean=(0.1307,), std=(0.3081,))
    return images, labels


def deepchecks_collate(model) -> t.Callable:
    """Process batch to deepchecks format.

    Parameters
    ----------
    model
        model to predict with
    Returns
    -------
    BatchOutputFormat
        batch of data in deepchecks format
    """

    def _process_batch_to_deepchecks_format(data) -> BatchOutputFormat:
        raw_images = torch.stack([x[0] for x in data])
        labels = [x[1] for x in data]
        predictions = model(raw_images)
        images = raw_images.permute(0, 2, 3, 1)
        images = un_normalize_batch(images, mean=(0.1307,), std=(0.3081,))
        return {'images': images, 'labels': labels, 'predictions': predictions}

    return _process_batch_to_deepchecks_format


def load_model(pretrained: bool = True, path: pathlib.Path = None,
               device: t.Union[str, torch.device] = 'cpu') -> 'MockModel':
    """Load MNIST model.

    Returns
    -------
    MnistModel
    """
    # TODO: should we put downloadable pre-trained model into our repo?
    if path and not path.exists():
        raise RuntimeError(f'Path for MNIST model not found: {str(path)}')

    path = path or MODEL_PATH
    dev = torch.device(device) if isinstance(device, str) else device
    if pretrained and path.exists():
        model = MnistModel()
        model.load_state_dict(torch.load(path))
        model.eval()
        return MockModel(model, dev)

    model = MnistModel()
    dataloader = t.cast(DataLoader, load_dataset(train=True, object_type='DataLoader'))
    datasize = len(dataloader.dataset)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 3

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
                LOGGER.info('Epoch: %f; loss=%f, %d/%d', epoch, loss, current, datasize)

    if not path.parent.exists():
        path.parent.mkdir()

    torch.save(model.state_dict(), path)
    model.eval()
    return MockModel(model, dev)


class MockModel:
    """Class of MNIST model that returns cached predictions."""

    def __init__(self, real_model, device):
        self.device = device
        self.real_model = real_model.to(device)
        with open(MNIST_DIR / 'static_predictions.pickle', 'rb') as handle:
            predictions = pickle.load(handle)
        self.cache = {key: torch.tensor(value).to(device) for key, value in predictions.items()}

    def __call__(self, batch):
        results = []
        for img in batch:
            hash_key = hash_image(img)
            if hash_key not in self.cache:
                prediction = self.real_model(torch.stack([img]).to(self.device))[0]
                prediction = nn.Softmax(dim=0)(prediction).detach()
                self.cache[hash_key] = prediction.to(self.device)
            results.append(self.cache[hash_key])
        return torch.stack(results)


class IterableTorchMnistDataset(IterableDataset):
    """Iterable MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Parameters
    ----------
    batch_size: int, default=64
        Batch size to use
    train: bool, default: true
        If True, creates dataset from ``training.pt``, otherwise from ``test.pt``.
    transform: t.Optional[t.Callable], default: None
        A function/transform that  takes in an PIL image and returns a transformed version.
        E.g, ``transforms.RandomCrop``
    n_samples: int, default: None
        Number of samples to use. If None, use all samples.
    """

    mirrors = ['http://yann.lecun.com/exdb/mnist/', 'https://ossci-datasets.s3.amazonaws.com/mnist/', ]

    resources = [('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
                 ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
                 ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
                 ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')]

    def __init__(self, batch_size: int = 64, train: bool = True, transform: t.Optional[t.Callable] = None,
                 n_samples: int = None) -> None:
        super().__init__()
        self.train = train  # training set or test set
        self.transform = transform
        self.batch_size = batch_size

        self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.data, self.targets = self._load_data()
        if n_samples:
            self.data = self.data[:n_samples]
            self.targets = self.targets[:n_samples]
        if self.transform is not None:
            self.data = torch.stack([self.transform(image=object_to_numpy(img))['image'] for img in self.data])

    def __iter__(self):
        """Iterate over the dataset."""
        return cycle(zip(self.data, self.targets))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    @property
    def raw_folder(self) -> str:
        """Return the path to the raw data folder."""
        return os.path.join(MNIST_DIR, 'raw_data')

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0])) for url, _ in
            self.resources)

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""
        if self._check_exists():
            return

        os.makedirs(MNIST_DIR, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f'{mirror}{filename}'
                try:
                    print(f'Downloading {url}')
                    download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
                except URLError as err:
                    get_logger().warning('Failed to download (trying next):\n%s', err)
                    continue
                break
            else:
                raise RuntimeError(f'Error downloading {filename}')


class TorchMnistDataset(datasets.MNIST):
    """MNIST Dataset."""

    @property
    def raw_folder(self) -> str:
        """Return the path to the raw data folder."""
        return os.path.join(self.root, 'raw_data')

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


class MnistModel(nn.Module):
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
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            return F.log_softmax(x, dim=1)
