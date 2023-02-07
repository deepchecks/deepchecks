"""
.. _vision_classification_tutorial:

==============================================
Image Classification Tutorial
==============================================

In this tutorial, you will learn how to validate your **classification model** using deepchecks test suites.
You can read more about the different checks and suites for computer vision use cases at the
:doc:`examples section  </checks_gallery/vision>`.

A classification model is usually used to classify an image into one of a number of classes. Although there are
multi label use-cases, in which the model is used to classify an image into multiple classes, most use-cases
require the model to classify images into a single class.
Currently, deepchecks supports only single label classification (either binary or multi-class).

.. code-block:: bash

    # Before we start, if you don't have deepchecks vision package installed yet, run:
    import sys
    !{sys.executable} -m pip install "deepchecks[vision]" --quiet --upgrade # --user

    # or install using pip from your python environment
"""

#%%
# Defining the data and model
# ===========================
# .. note::
#   In this tutorial, we use the pytorch to create the dataset and model. To see how this can be done using tensorflow
#   or other frameworks, please visit the :ref:`creating VisionData guide <vision_data_class>`.

#%%
# Downloading the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~
# The data is available from the torch library. We will download and extract it to the current directory.

import os
import urllib.request
import zipfile

url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
urllib.request.urlretrieve(url, './hymenoptera_data.zip')

with zipfile.ZipFile('./hymenoptera_data.zip', 'r') as zip_ref:
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

import albumentations as A
import numpy as np
import PIL.Image
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader

class AntsBeesDataset(torchvision.datasets.ImageFolder):

    def __getitem__(self, index: int):
        """overrides __getitem__ to be compatible to albumentations"""
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
            return np.array(image).astype('uint8')
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise RuntimeError("Only PIL.Image and CV2 loaders currently supported!")

data_dir = './hymenoptera_data'
# Just normalization for validation
data_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
train_dataset = AntsBeesDataset(root=os.path.join(data_dir,'train'))
train_dataset.transforms = data_transforms

test_dataset = AntsBeesDataset(root=os.path.join(data_dir, 'val'))
test_dataset.transforms = data_transforms

#%%
# Visualize the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's see how our data looks like.

print(f'Number of training images: {len(train_dataset)}')
print(f'Number of validation images: {len(test_dataset)}')
print(f'Example output of an image shape: {train_dataset[0][0].shape}')
print(f'Example output of a label: {train_dataset[0][1]}')

#%%
# Downloading a pre-trained model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now, we will download a pre-trained model from torchvision, that was trained on the ImageNet dataset.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
# Implementing the VisionData class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The checks in the package validate the model & data by calculating various quantities over the data, labels and
# predictions. In order to do that, those must be in a pre-defined format, according to the task type.
# In the following example we're using pytorch. To see an implementation of this in tensorflow, please refer to
# :ref:`creating VisionData guide <vision_data_class>`
# For pytorch, we will use our DataLoader, but we'll create a new collate function for it, that transforms the batch to
# the correct format. Then, we'll create a :class:`deepchecks.vision.vision_data.vision_data.VisionData` object, that will hold the data loader.
#
# To learn more about the expected formats, please visit the
# :doc:`supported tasks and formats guide </user-guide/vision/supported_tasks_and_formats>`.
#
# First, we'll create the collate function that will be used by the DataLoader.
# In pytorch, the collate function is used to transform the output batch to any custom format, and we'll use that
# in order to transform the batch to the correct format for the checks.

from deepchecks.vision.vision_data import BatchOutputFormat

def deepchecks_collate_fn(batch) -> BatchOutputFormat:
    """Return a batch of images, labels and predictions for a batch of data. The expected format is a dictionary with
    the following keys: 'images', 'labels' and 'predictions', each value is in the deepchecks format for the task.
    You can also use the BatchOutputFormat class to create the output.
    """
    # batch received as iterable of tuples of (image, label) and transformed to tuple of iterables of images and labels:
    batch = tuple(zip(*batch))

    # images:
    inp = torch.stack(batch[0]).detach().numpy().transpose((0, 2, 3, 1))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp = std * inp + mean
    images = np.clip(inp, 0, 1) * 255

    #labels:
    labels = batch[1]

    #predictions:
    logits = model.to(device)(torch.stack(batch[0]).to(device))
    predictions = nn.Softmax(dim=1)(logits)
    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)

#%%
# We have a single label here, which is the tomato class
# The label_map is a dictionary that maps the class id to the class name, for display purposes.

LABEL_MAP = {
    0: 'ants',
    1: 'bees'
  }

#%%
# Now that we have our updated collate function, we can recreate the dataloader in the deepchecks format, and use it
# to create a VisionData object:

from deepchecks.vision import VisionData

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=deepchecks_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=deepchecks_collate_fn)

training_data = VisionData(batch_loader=train_loader, task_type='classification', label_map=LABEL_MAP)
test_data = VisionData(batch_loader=test_loader, task_type='classification', label_map=LABEL_MAP)
#%%
# Making sure our data is in the correct format:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The VisionData object automatically validates your data format and will alert you if there is a problem.
# However, you can also manually view your images and labels to make sure they are in the correct format by using
# the ``head`` function to conveniently visualize your data:

training_data.head()

#%%
# And observe the output:
#
# Running Deepchecks' suite on our data and model!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now that we have defined the task class, we can validate the train and test data with deepchecks' train test validation
# suite.
# This can be done with this simple few lines of code:

from deepchecks.vision.suites import train_test_validation

suite = train_test_validation()
result = suite.run(training_data, test_data,  max_samples = 5000)

#%%
# We also have suites for:
# :func:`data integrity <deepchecks.vision.suites.data_integrity>`
# - validating a single dataset and
# :func:`model evaluation <deepchecks.vision.suites.model_evaluation>` -
# evaluating the model's performance.

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

#%%
# We can see that we do not have any meaningful issues with our data, and although there's some drift between the
# train and test datasets (under the "Passed" section), this is not significant enough to cause any issues (and therefor
# is not displayed in the "Didn't Pass" section).
# However, under the "Other" section, that details checks without a specific pass/fail condition, we can see that the
# heatmap of brightness in the images is not uniformly distributed, which means that in most images, there are brighter
# objects in the center of the image. This makes sense as these images of bees and ants tend to have the insects in the
# center of the image, but it is something to be aware of and maybe use data augmentation to fix.