# -*- coding: utf-8 -*-
"""
Image Dataset Drift
===================
This notebooks provides an overview for using and understanding the image dataset
drift check, used to detect drift in simple image properties between train and
test datasets.

**Structure:**

* `What Is Dataset Drift? <#what-is-dataset-drift>`__
* `How Does the ImageDatasetDrift Check Work? <#how-does-the-imagedatasetdrift-check-work>`__
* `Which Image Properties Are Used? <#which-image-properties-are-used>`__
* `Loading The Data <#loading-the-data>`__
* `Run The Check <#run-the-check>`__


What Is Dataset Drift?
------------------------
Data drift is simply a change in the distribution of data over time. It is also
one of the top reasons that a machine learning model performance degrades over time.

Specifically, a whole dataset drift, or a multivariate dataset drift, occurs when
there is a change in the relation between input features.

Causes of data drift include:

* Natural drift in the data, such as lighting (brightness) changes between summer
  and winter.
* Upstream process changes, such as a camera being replaced that has a different
  lens, which makes images sharper.
* Data quality issues, such as a malfunctioning camera that always returns a black image.
* Data pipeline errors, such as a change in image augmentations done in preprocessing.

In the context of machine learning, drift between the training set and the test set
(which is not due to augmentation) will likely make the model prone to errors. In
other words, if the model was trained on data that is different from the current test
data, it will probably make more mistakes predicting the target variable.

How Does the ImageDatasetDrift Check Work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are many methods to detect feature drift. Some of them are statistical methods
that aim to measure difference between distribution of 2 given sets. This methods
are more suited to univariate distributions and are primarily used to detect drift
between 2 subsets of a single feature.

Measuring a multivariate data drift is a bit more challenging. In the image dataset
drift check, the multivariate drift is measured by training a classifier that detects
which samples come from a known distribution and defines the drift by the accuracy
of this classifier.

Practically, the check concatenates the train and the test sets, and assigns label 0
to samples that come from the training set, and 1 to those from the test set.
Then, we train a binary classifer of type `Histogram-based Gradient Boosting
Classification Tree <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>`_, 
and measure the drift score from the AUC score of this classifier.

As the classifier is a tree model, that cannot run on the images themselves, the
check calculates properties for each image (such as brightness, aspect ratio etc.)
and uses them as input features to the classifier.

Which Image Properties Are Used?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
==============================  ==========
Property name                   What is it
==============================  ==========
Aspect Ratio                    Ratio between height and width of image (height / width)
Area                            Area of image in pixels (height * width)
Brightness                      Average intensity of image pixels. Color channels have different weights according to
                                RGB-to-Grayscale formula
RMS Contrast                    Contrast of image, calculated by standard deviation of pixels
Mean Red Relative Intensity     Mean over all pixels of the red channel, scaled to their relative intensity in
                                comparison to the other channels [r / (r + g + b)].
Mean Green Relative Intensity   Mean over all pixels of the green channel, scaled to their relative intensity in
                                comparison to the other channels [g / (r + g + b)].
Mean Blue Relative Intensity    Mean over all pixels of the blue channel, scaled to their relative intensity in
                                comparison to the other channels [b / (r + g + b)].
==============================  ==========
"""

#%%
# Imports
# -------

import numpy as np

from deepchecks.vision.checks import ImageDatasetDrift

#%%
# Loading the data
# ----------------

from deepchecks.vision.datasets.detection.coco import load_dataset

train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')

#%%
# Run the check
# -------------
# without drift
# ^^^^^^^^^^^^^

check = ImageDatasetDrift()
check.run(train_dataset=train_ds, test_dataset=test_ds)

#%%
# Insert drift
# ^^^^^^^^^^^^
# Now, we will define a custom data object that will insert a drift to the training set.

from deepchecks.vision.datasets.detection.coco import COCOData

def add_brightness(img):
    reverse = 255 - img
    addition_of_brightness = (reverse * 0.07).astype(int)
    return img + addition_of_brightness


class DriftedCOCO(COCOData):
    
    def batch_to_images(self, batch):
        return [add_brightness(np.array(img)) for img in batch[0]]

#%%
train_dataloader = load_dataset(train=True, object_type='DataLoader')
test_dataloader = load_dataset(train=False, object_type='DataLoader')

drifted_train_ds = DriftedCOCO(train_dataloader)
test_ds = COCOData(test_dataloader)

#%%
# Run the check again
# ^^^^^^^^^^^^^^^^^^^
check = ImageDatasetDrift()
check.run(train_dataset=drifted_train_ds, test_dataset=test_ds)
