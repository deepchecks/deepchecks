import json
import pickle
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch

from deepchecks.vision.datasets.classification import mnist
# mean = (0.1307,)
# std = (0.3081,)
# transform = A.Compose([A.Normalize(mean, std), ToTensorV2(), ])
# dataset = mnist.IterableTorchMnistDataset(train=True, transform=transform)
data_loader = mnist.load_dataset(object_type='DataLoader', use_iterable_dataset=True)
data_loader_regular = mnist.load_dataset(object_type='DataLoader')
print('end')