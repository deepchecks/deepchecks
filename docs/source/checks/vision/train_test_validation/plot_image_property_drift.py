# -*- coding: utf-8 -*-
"""
.. _plot_vision_image_property_drift:

Image Property Drift
********************

This notebooks provides an overview for using and understanding the image property drift check.

**Structure:**

* `What Is Image Drift? <#what-is-image-drift>`__
* `Which Image Properties Are Used? <#which-image-properties-are-used>`__
* `Prepare data <#prepare-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__
* `Check Parameters <#check-parameters>`__

What Is Image Drift?
=================================
Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Image drift is a data drift that occurs in images in the dataset.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide>`.

How Deepchecks Detects Image Drift
------------------------------------

This check detects image property drift by using :ref:`univariate measures <drift_detection_by_univariate_measure>`
on each image property separately.
Another possible method for drift detection is by :ref:`a domain classifier <drift_detection_by_domain_classifier>`
which is used in the :doc:`Image Dataset Drift check </checks_gallery/vision/train_test_validation/plot_image_dataset_drift>`.

Using Properties to Detect Image Drift
--------------------------------------------
In computer vision specifically, we can't measure drift on images directly, as the individual pixel has little
value when estimating drift. Therefore, we calculate drift on different :doc:`properties of the image</user-guide/vision/vision_properties>`,
on which we can directly measure drift.


Which Image Properties Are Used?
=================================
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

Imports
-------
"""

#%%
#
# .. note::
#   In this example, we use the pytorch version of the coco dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.detection import coco_tensorflow as coco

from deepchecks.vision.checks import ImagePropertyDrift
from deepchecks.vision.datasets.detection import coco_torch as coco
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
# To display the results in an IDE like PyCharm, you can use the following code:

#  check_result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# Observe the check’s output 
# --------------------------
# The result value is a pandas DataFrame that contains drift score for each image property.

check_result.value

#%%
# We can also pass the check a list of classes we wish to inspect, and the check will calculate the properties only
# for images either belonging to the classes or containing annotations belonging to the classes. (We'll lower the
# min_samples to 5 to tell the check to calculate drift despite having only a few images left after the class
# filtration)

check_result = ImagePropertyDrift(classes_to_display=['person', 'traffic light'], min_samples=5
                                  ).run(train_dataset, test_dataset)
check_result


#%%
# Define a condition
# ==================
# We can define a condition that make sure that image properties drift scores do not
# exceed allowed threshold.

check_result = (
    ImagePropertyDrift()
    .add_condition_drift_score_less_than(0.001)
    .run(train_dataset, test_dataset)
)
check_result.show(show_additional_outputs=False)

#%%
# Check Parameters
# ----------------
# Image Property Drift Check accepts two parameters that allows us to control the look of the output:
#
# * `vision_properties` - list of image properties that we are interested in
# * `max_num_categories` - Maximal number of categories to use for the calculation of drift using PSI (Population Stability Index)
#
# Only next string values are allowed for the `vision_properties` parameter:
#
# * `aspect_ratio`
# * `area`
# * `brightness`
# * `mean_red_relative_intensity`
# * `mean_green_relative_intensity`
# * `mean_blue_relative_intensity`

from typing import List

import numpy as np


def area(images: List[np.ndarray]) -> List[int]:
    # Return list of integers of image areas (height multiplied by width)
    return [img.shape[0] * img.shape[1] for img in images]


def aspect_ratio(images: List[np.ndarray]) -> List[float]:
    # Return list of floats of image height to width ratio
    return [img.shape[0] / img.shape[1] for img in images]


properties = [
    {'name': 'Area', 'method': area, 'output_type': 'numerical'},
    {'name': 'Aspect Ratio', 'method': aspect_ratio, 'output_type': 'numerical'}
]

check_result = ImagePropertyDrift(
    image_properties=properties,
    max_num_categories_for_drift=20
).run(train_dataset, test_dataset)

check_result