"""
.. _vision_segmentation_tutorial:

===============================
Semantic Segmentation Tutorial
===============================

In this tutorial, you will learn how to validate your **semantic segmentation model** using deepchecks test suites.
You can read more about the different checks and suites for computer vision use cases at the
:doc:`examples section </checks_gallery/vision/index>`.

If you just want to see the output of this tutorial, jump to :ref:`observing_the_result` section.

A semantic segmentation task is a task where every pixel of the image is labeled with a single class.
Therefore, a common output of these tasks is an image of identical size to the input, with a vector for each pixel
of the probability for each class.

.. code-block:: bash

    # Before we start, if you don't have deepchecks vision package installed yet, run:
    import sys
    !{sys.executable} -m pip install "deepchecks[vision]" --quiet --upgrade # --user

    # or install using pip from your python environment
"""

# %%
# Defining the data and model
# ===========================
# .. note::
#   In this tutorial, we use the pytorch to create the dataset and model. To see how this can be done using tensorflow
#   or other frameworks, please visit the :ref:`creating VisionData guide <vision_data_class>`.

# %%
# Load Data
# ~~~~~~~~~
# The model in this tutorial is used to detect different object segments in images (labels based on the Pascal VOC dataset).
# The model is trained to identify 20 different objects (person, bicycle etc.) and background.
# The dataset itself is the COCO128 dataset with semantic segmentation labels, mapped to the Pascal VOC labels
# (Originally, the COCO dataset includes more labels, but those have been filtered out).
# The dataset can be loaded as a pytorch Dataset object from deepchecks.vision.datasets.segmentation, as is done in
# this tutorial, but can also be loaded as a VisionData object using the "load_dataset" function from that directory,

# The full pascal VOC data and information can be found here: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
# And the COCO128 dataset can be found here: https://www.kaggle.com/datasets/ultralytics/coco128

from deepchecks.vision.datasets.segmentation.segmentation_coco import CocoSegmentationDataset, load_model

train_dataset = CocoSegmentationDataset.load_or_download(train=True)
test_dataset = CocoSegmentationDataset.load_or_download(train=False)

#%%
# Visualize the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's see how our data looks like.

print(f'Number of training images: {len(train_dataset)}')
print(f'Number of test images: {len(test_dataset)}')
print(f'Example output of an image shape: {train_dataset[0][0].shape}')
print(f'Example output of a label shape: {train_dataset[0][1].shape}')

# %%
# Downloading a Pre-trained Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this tutorial, we will download a pre-trained LRSAPP model and a MobileNetV3 Large backbone
# from the official PyTorch repository. For more details, please refer to the
# `official documentation <https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html>`__.

model = load_model(pretrained=True)

#%%
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

import torch
import torchvision.transforms.functional as F
from deepchecks.vision.vision_data import BatchOutputFormat

def deepchecks_collate_fn(batch) -> BatchOutputFormat:
    """Return a batch of images, labels and predictions for a batch of data. The expected format is a dictionary with
    the following keys: 'images', 'labels' and 'predictions', each value is in the deepchecks format for the task.
    You can also use the BatchOutputFormat class to create the output.
    """
    # batch received as iterable of tuples of (image, label) and transformed to tuple of iterables of images and labels:
    batch = tuple(zip(*batch))

    # images:
    images = [tensor.numpy().transpose((1, 2, 0)) for tensor in batch[0]]

    #labels:
    labels = batch[1]

    #predictions:
    normalized_batch = [F.normalize(img.unsqueeze(0).float() / 255,
                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for img in batch[0]]
    predictions = [model(img)["out"].squeeze(0).detach() for img in normalized_batch]
    predictions = [torch.nn.functional.softmax(pred, dim=0) for pred in predictions]

    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)

# %%
# The label_map is a dictionary that maps the class id to the class name, for display purposes.
LABEL_MAP = {0: 'background', 1: 'airplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car',
             8: 'cat', 9: 'chair', 10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse', 14: 'motorcycle',
             15: 'person', 16: 'potted plant', 17: 'sheep', 18: 'couch', 19: 'train', 20: 'tv'}

#%%
# Now that we have our updated collate function, we can create the dataloader in the deepchecks format, and use it
# to create a VisionData object:

from torch.utils.data import DataLoader
from deepchecks.vision import VisionData

train_loader = DataLoader(dataset=train_dataset, shuffle=True, collate_fn=deepchecks_collate_fn)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, collate_fn=deepchecks_collate_fn)

training_data = VisionData(batch_loader=train_loader, task_type='semantic_segmentation', label_map=LABEL_MAP)
test_data = VisionData(batch_loader=test_loader, task_type='semantic_segmentation', label_map=LABEL_MAP)

#%%
# Making sure our data is in the correct format:
# ~~~~~~~~~~~~~~~~~~~~~~
# The VisionData object automatically validates your data format and will alert you if there is a problem.
# However, you can also manually view your images and labels to make sure they are in the correct format by using
# the ``head`` function to conveniently visualize your data:

training_data.head()

# %%
# Running Deepchecks' model evaluation suite on our data and model!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now that we have defined the task class, we can validate the model with the model evaluation suite of deepchecks.
# This can be done with this simple few lines of code:

from deepchecks.vision.suites import model_evaluation

suite = model_evaluation()
result = suite.run(training_data, test_data)

# %%
# .. _vision_segmentation_tutorial__observing_the_result:
#
# Observing the results:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The results can be saved as a html file with the following code:

result.save_as_html('output.html')

# %%
# Or, if working inside a notebook, the output can be displayed directly by simply printing the result object:

result.show()

# %%
# From these results, we can see that mostly our model performs well. However, the model had an issue with identifying
# a specific class ("bicycle") in the test set, which casued a major degradation in the
# `Dice metric <https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient>`_ for that class,
# as can be seen in the check "Class Performance" under the "Didn't Pass" section.
# However, as this dataset has very few samples, this would require further investigation.
#
# We can also see that there are significant changes between the train and test set, regarding the model's predictions
# on them. in the "Train Test Prediction Drift" check, which checks drift in 3 properties of the predictions, we can
# see there's a change in the distribution of the predicted classes.
# This can tell us that the train set is not representing the test set well, even without knowing the actual test set
# labels.
