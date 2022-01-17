import pytest
import torch
from torch import nn
from torchvision.transforms import ToTensor

from deepchecks.vision import VisionDataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


@pytest.fixture(scope='session')
def mnist_data_loader():
    mnist_train_dataset = MNIST('./mnist',
                                download=True,
                                train=True,
                                transform=ToTensor())

    return DataLoader(mnist_train_dataset, batch_size=64)


@pytest.fixture(scope='session')
def mnist_dataset(mnist_data_loader):
    """Return Iris dataset modified to a binary label as Dataset object."""
    dataset = VisionDataset(mnist_data_loader)
    return dataset


@pytest.fixture(scope='session')
def simple_nn():
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
def trained_mnist(simple_nn, mnist_data_loader):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(simple_nn.parameters(), lr=1e-3)
    size = len(mnist_data_loader.dataset)
    # Training 1 epoch
    simple_nn.train()
    for batch, (X, y) in enumerate(mnist_data_loader):
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
def ssd_utils():
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils', map_location=torch.device('cpu'))

    return utils
@pytest.fixture(scope='session')
def trained_ssd_object_detection():
    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', map_location=torch.device('cpu'))

    ssd_model.to('cpu')
    ssd_model.eval()

    return ssd_model

@pytest.fixture(scope='session')
def obj_detection_images(ssd_utils):
    uris = [
        'http://images.cocodataset.org/val2017/000000397133.jpg',
        'http://images.cocodataset.org/val2017/000000037777.jpg',
        'http://images.cocodataset.org/val2017/000000252219.jpg'
    ]

    inputs = [ssd_utils.prepare_input(uri) for uri in uris]
    tensor = ssd_utils.prepare_tensor(inputs)

    return tensor