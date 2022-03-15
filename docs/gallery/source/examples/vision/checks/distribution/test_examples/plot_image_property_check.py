# -*- coding: utf-8 -*-
"""
Image Property Drift Check
**************************
This notebooks provides an overview for using and understanding the image property drift check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Prepare data <#prepare-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__
* `Check Parameters <#check-parameters>`__

What is the purpose of the check?
=================================
Data drift is simply a change in the distribution of data over time. It is also one
of the top reasons of a machine learning model performance degrades over time.

In the context of machine learning, drift between the training set and the test set
will likely make the model to be prone to errors. In other words, this means that
the model was trained on data that is different from the current test data, thus
it will probably make more mistakes predicting the target variable.

Image Property Drift check calculates a drift score for each image property in
test dataset, by comparing its distribution to the train dataset. For this, we
use the Earth Movers Distance (Wasserstein distance).

Imports
-------
"""

#%%

from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.checks.distribution import ImagePropertyDrift

#%%
# Prepare data
# ------------
from deepchecks.vision.utils import image_properties

train_dataset = coco.load_dataset(train=True, object_type='VisionData')
test_dataset = coco.load_dataset(train=False, object_type='VisionData')

#%%
# Run the check 
# -------------

check_result = ImagePropertyDrift().run(train_dataset, test_dataset)
check_result

#%%
# Observe the check’s output 
# --------------------------
# The result value is a pandas DataFrame that contains drift score for each image property.

check_result.value

#%%
# Define a condition
# ==================
# We can define a condition that make sure that image properties drift scores do not
# exceed allowed threshold.

check_result = (
    ImagePropertyDrift()
    .add_condition_drift_score_not_greater_than(0.001)
    .run(train_dataset, test_dataset)
)
check_result.show(show_additional_outputs=False)

#%%
# Check Parameters
# ----------------
# Image Property Drift Check accepts two parameters that allows us to control the look of the output:
#
# * `image_properties` - list of image properties that we are interested in
# * `max_num_categories` - Maximal number of categories to use for the calculation of drift using PSI (Population Stability Index)
#
# Only next string values are allowed for the `image_properties` parameter:
#
# * `aspect_ratio`
# * `area`
# * `brightness`
# * `normalized_red_mean`
# * `normalized_green_mean`
# * `normalized_blue_mean`

from typing import List
import numpy as np


def area(images: List[np.ndarray]) -> List[int]:
    # Return list of integers of image areas (height multiplied by width)
    return [img.shape[0] * img.shape[1] for img in images]


def aspect_ratio(images: List[np.ndarray]) -> List[float]:
    # Return list of floats of image height to width ratio
    return [img.shape[0] / img.shape[1] for img in images]


properties = [
    {'name': 'Area', 'method': area, 'output_type': 'continuous'},
    {'name': 'Aspect Ratio', 'method': aspect_ratio, 'output_type': 'continuous'}
]

check_result = ImagePropertyDrift(
    alternative_image_properties=properties,
    max_num_categories=20
).run(train_dataset, test_dataset)

check_result