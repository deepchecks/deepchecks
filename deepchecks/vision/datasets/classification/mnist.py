import os.path

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision


import pathlib
current_path = pathlib.Path(__file__).parent.resolve()
model_path = str(current_path).replace('\\', '/') + '/models/mnist.pth'


class MNistNet(nn.Module):
    def __init__(self):
        super(MNistNet, self).__init__()
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

batch_size_train = 64
batch_size_test = 1000

mnist_data_path = os.path.join(current_path, '../../../datasets/classification/mnist')

mnist_data = torchvision.datasets.MNIST(mnist_data_path, download=True)
mnist_train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(mnist_data_path, train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True, pin_memory=True)

mnist_test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(mnist_data_path, train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True, pin_memory=True)

def load_mnist():
    model = torch.load(model_path)
    model.eval()

    return model