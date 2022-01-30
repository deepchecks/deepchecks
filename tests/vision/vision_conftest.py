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
import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from deepchecks.vision import VisionDataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from deepchecks.vision.datasets.detection.coco import get_trained_yolov5_object_detection, get_coco_dataloader


@pytest.fixture(scope='session')
def mnist_data_loader_train():
    mnist_train_dataset = MNIST('./mnist',
                                download=True,
                                train=True,
                                transform=ToTensor())

    return DataLoader(mnist_train_dataset, batch_size=64)


@pytest.fixture(scope='session')
def mnist_dataset_train(mnist_data_loader_train):
    """Return MNist dataset as VisionDataset object."""
    dataset = VisionDataset(mnist_data_loader_train)
    return dataset


@pytest.fixture(scope='session')
def mnist_data_loader_test():
    mnist_train_dataset = MNIST('./mnist',
                                download=True,
                                train=False,
                                transform=ToTensor())

    return DataLoader(mnist_train_dataset, batch_size=1000)


@pytest.fixture(scope='session')
def mnist_dataset_test(mnist_data_loader_test):
    """Return MNist dataset as VisionDataset object."""
    dataset = VisionDataset(mnist_data_loader_test)
    return dataset


@pytest.fixture(scope='session')
def simple_nn():
    torch.manual_seed(42)

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to('cpu')
    return model


@pytest.fixture(scope='session')
def trained_mnist(simple_nn, mnist_data_loader_train):
    torch.manual_seed(42)
    simple_nn = copy.deepcopy(simple_nn)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(simple_nn.parameters(), lr=1e-3)
    size = len(mnist_data_loader_train.dataset)
    # Training 1 epoch
    simple_nn.train()
    for batch, (X, y) in enumerate(mnist_data_loader_train):
        X, y = X.to('cpu'), y.to('cpu')

        # Compute prediction error
        pred = simple_nn(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return simple_nn


@pytest.fixture(scope='session')
def trained_yolov5_object_detection():
    return get_trained_yolov5_object_detection()


@pytest.fixture(scope='session')
def obj_detection_images():
    uris = [
        'http://images.cocodataset.org/val2017/000000397133.jpg',
        'http://images.cocodataset.org/val2017/000000037777.jpg',
        'http://images.cocodataset.org/val2017/000000252219.jpg'
    ]

    return uris


@pytest.fixture(scope='session')
def coco_dataloader():
    return get_coco_dataloader(batch_size=4)


@pytest.fixture(scope='session')
def coco_dataset(coco_dataloader):
    return VisionDataset(coco_dataloader)