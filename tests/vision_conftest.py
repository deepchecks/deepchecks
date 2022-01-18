import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from deepchecks.vision import VisionDataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST



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
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    model = NeuralNetwork().to('cpu')
    return model


@pytest.fixture(scope='session')
def trained_mnist(simple_nn, mnist_data_loader_train):
    torch.manual_seed(42)
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
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()

    return model

@pytest.fixture(scope='session')
def obj_detection_images():
    uris = [
        'http://images.cocodataset.org/val2017/000000397133.jpg',
        'http://images.cocodataset.org/val2017/000000037777.jpg',
        'http://images.cocodataset.org/val2017/000000252219.jpg'
    ]

    return uris