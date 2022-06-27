# -*- coding: utf-8 -*-
"""
Image Dataset Drift
*********************

This notebooks provides an overview for using and understanding the image dataset
drift check, used to detect drift in simple image properties between train and
test datasets.

**Structure:**

* `What Is Image Dataset Drift? <#what-is-image-dataset-drift>`__
* `Which Image Properties Are Used? <#which-image-properties-are-used>`__
* `Loading The Data <#loading-the-data>`__
* `Run The Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

What Is Image Dataset Drift?
------------------------------------

Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Image dataset drift is a drift that occurs in more than one image property at a time,
and may even affect the relationships between those properties, which are undetectable by
univariate drift methods.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide>`.

How Deepchecks Detects Dataset Drift
------------------------------------

This check detects multivariate drift by using :ref:`a domain classifier <drift_detection_by_domain_classifier>`.
Other methods to detect drift include :ref:`univariate measures <drift_detection_by_univariate_measure>`
which is used in other checks, such as :doc:`Image Property Drift check </checks_gallery/vision/train_test_validation/plot_image_property_drift>`.

Using Properties to Detect Image Drift
--------------------------------------------
In computer vision specifically, we can't measure drift on the images directly, as the individual pixel has little
value when estimating drift. Therefore, we calculate drift on different :doc:`properties of the image</user-guide/vision/vision_properties>`,
on which we can directly measure drift.

Which Image Properties Are Used?
------------------------------------
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
from deepchecks.vision.datasets.detection.coco import load_dataset

#%%
# Loading the data
# ----------------


train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')

#%%
# Run the check
# -------------
# without drift
# ^^^^^^^^^^^^^

check = ImageDatasetDrift()
result = check.run(train_dataset=train_ds, test_dataset=test_ds)
result

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.

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
test_ds_coco = COCOData(test_dataloader)

#%%
# Run the check again
# ^^^^^^^^^^^^^^^^^^^
check = ImageDatasetDrift()
result = check.run(train_dataset=drifted_train_ds, test_dataset=test_ds_coco)
result

#%%
# Define a Condition
# -------------
# Now, we will define a condition that the maximum drift score is less than a certain threshold. In this example we will
# set the threshold at 0.2.
# In order to demonstrate the condition, we will use again the original (not drifted) train dataset.

check = ImageDatasetDrift().add_condition_drift_score_less_than(0.2)
result = check.run(train_dataset=train_ds, test_dataset=test_ds).show(show_additional_outputs=False)
result
