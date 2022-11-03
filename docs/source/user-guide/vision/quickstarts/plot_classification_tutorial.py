"""
.. _vision_classification_tutorial:

==============================================
Classification Model Validation Tutorial
==============================================

In this tutorial, you will learn how to validate your **classification model** using deepchecks test suites.
You can read more about the different checks and suites for computer vision use cases at the
:doc:`examples section  </checks_gallery/vision>`.

A classification model is usually used to classify an image into one of a number of classes. Although there are
multi label use-cases, in which the model is used to classify an image into multiple classes, most use-cases
require the model to classify images into a single class.
Currently deepchecks supports only single label classification (either binary or multi-class).

.. code-block:: bash

    # Before we start, if you don't have deepchecks vision package installed yet, run:
    import sys
    !{sys.executable} -m pip install "deepchecks[vision]" --quiet --upgrade # --user

    # or install using pip from your python environment
"""

#%%
# Defining the data and model
# ===========================

# Importing the required packages

import os
import urllib.request
import zipfile

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

from deepchecks.vision.classification_data import ClassificationData

#%%
# Downloading the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~
# The data is available from the torch library. We will download and extract it to the current directory.

url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
urllib.request.urlretrieve(url, 'hymenoptera_data.zip')

with zipfile.ZipFile('hymenoptera_data.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
#%%
# Load Data
# ~~~~~~~~~
#
# We will use torchvision and torch.utils.data packages for loading the data.
# The model we are building will learn to classify **ants** and **bees**.
# We have about 120 training images each for ants and bees.
# There are 75 validation images for each class.
# This dataset is a very small subset of imagenet.

class AntsBeesDataset(ImageFolder):
    def __init__(self, *args, **kwargs):
        """
        Overrides initialization method to replace default loader with OpenCV loader
        :param args:
        :param kwargs:
        """
        super(AntsBeesDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        """
        overrides __getitem__ to be compatible to albumentations
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.get_cv2_image(sample)
        if self.transforms is not None:
            transformed = self.transforms(image=sample, target=target)
            sample, target = transformed["image"], transformed["target"]
        else:
            if self.transform is not None:
                sample = self.transform(image=sample)['image']
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target

    def get_cv2_image(self, image):
        if isinstance(image, PIL.Image.Image):
            image_np = np.array(image).astype('uint8')
            return image_np
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise RuntimeError("Only PIL.Image and CV2 loaders currently supported!")

# Just normalization for validation
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = 'hymenoptera_data'
# Just normalization for validation
data_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
train_dataset = AntsBeesDataset(root=os.path.join(data_dir,'train'))
train_dataset.transforms = data_transforms

val_dataset = AntsBeesDataset(root=os.path.join(data_dir,'val'))
val_dataset.transforms = data_transforms

dataloaders = {
    'train':torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                                shuffle=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=4,
                                                shuffle=True)
}

class_names = ['ants', 'bees']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
# Visualize a Few Images
# ~~~~~~~~~~~~~~~~~~~~~~
# Let's visualize a few training images so as to understand the data augmentation.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

#%%
# .. image :: /_static/images/tutorials/ants-bees.png
#   :width: 400
#   :alt: Ants and Bees
#
# Downloading a pre-trained model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now, we will download a pre-trained model from torchvision, that was trained on the ImageNet dataset.

model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# We have only 2 classes
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
_ = model.eval()

#%%
# Validating the Model with Deepchecks
# =====================================
# Now, after we have the training data, validation data and the model, we can validate the model with
# deepchecks test suites.
# 
# Visualize the data loader and the model outputs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# First we'll make sure we are familiar with the data loader and the model outputs.

batch = next(iter(dataloaders['train']))

print("Batch type is: ", type(batch))
print("First element is: ", type(batch[0]), "with len of ", len(batch[0]))
print("Example output of an image shape from the dataloader ", batch[0][0].shape)
print("Image values", batch[0][0])
print("-"*80)

print("Second element is: ", type(batch[1]), "with len of ", len(batch[1]))
print("Example output of a label shape from the dataloader ", batch[1][0].shape)
print("Image values", batch[1][0])

#%%
# Implementing the ClassificationData class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The first step is to implement a class that enables deepchecks to interact with your model and data.
# The appropriate class to implement should be selected according to you models task type. In this tutorial,
# we will implement the classification task type by implementing a class that inherits from the
# :class:`deepchecks.vision.classification_data.ClassificationData` class.

# The goal of this class is to make sure the outputs of the model and of the dataloader are in the correct format.
# To learn more about the expected format please visit the API reference for the
# :class:`deepchecks.vision.classification_data.ClassificationData` class.

class AntsBeesData(ClassificationData):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    def batch_to_images(self, batch):
        """
        Convert a batch of data to images in the expected format. The expected format is an iterable of cv2 images,
        where each image is a numpy array of shape (height, width, channels). The numbers in the array should be in the
        range [0, 255]
        """
        inp = batch[0].detach().numpy().transpose((0, 2, 3, 1))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp*255

    def batch_to_labels(self, batch):
        """
        Convert a batch of data to labels in the expected format. The expected format is a tensor of shape (N,),
        where N is the number of samples. Each element is an integer representing the class index.
        """
        return batch[1]

    def infer_on_batch(self, batch, model, device):
        """
        Returns the predictions for a batch of data. The expected format is a tensor of shape (N, n_classes),
        where N is the number of samples. Each element is an array of length n_classes that represent the probability of
        each class.
        """
        logits = model.to(device)(batch[0].to(device))
        return nn.Softmax(dim=1)(logits)

#%%
# After defining the task class, we can validate it by running the following code:

LABEL_MAP = {
    0: 'ants',
    1: 'bees'
  }
training_data = AntsBeesData(data_loader=dataloaders["train"], label_map=LABEL_MAP)
val_data = AntsBeesData(data_loader=dataloaders["val"], label_map=LABEL_MAP)

training_data.validate_format(model)
val_data.validate_format(model)

#%%
# And observe the output:
#
# Running Deepchecks' full suite on our data and model!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now that we have defined the task class, we can validate the train and test data with the model train test validation
# suite of deepchecks.
# This can be done with this simple few lines of code:

from deepchecks.vision.suites import train_test_validation

suite = train_test_validation()
result = suite.run(training_data, val_data, model, device=device)

#%%
# Observing the results:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The results can be saved as a html file with the following code:

result.save_as_html('output.html')

# Or displayed in a new window in an IDE like Pycharm:
# result.show_in_window()

#%%
# Or, if working inside a notebook, the output can be displayed directly by simply printing the result object:

result
