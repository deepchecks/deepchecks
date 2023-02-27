# -*- coding: utf-8 -*-
"""
.. _plot_vision_image_property_outliers:

Image Property Outliers
=======================

This notebooks provides an overview for using and understanding the image property
outliers check, used to detect outliers in simple image properties in a dataset.

**Structure:**

* `Why Check for Outliers? <#why-check-for-outliers>`__
* `How Does the Check Work? <#how-does-the-check-work>`__
* `Which Image Properties Are Used? <#which-image-properties-are-used>`__
* `Run the Check <#run-the-check>`__


Why Check for Outliers?
-----------------------
Examining outliers may help you gain insights that you couldn't have reached from taking an aggregate look or by
inspecting random samples. For example, it may help you understand you have some corrupt samples (e.g.
an image that is completely black), or samples you didn't expect to have (e.g. extreme aspect ratio).
In some cases, these outliers may help debug some performance discrepancies (the model can be excused for failing on
a totally dark image). In more extreme cases, the outlier samples may indicate the presence of samples interfering with
the model's training by teaching the model to fit "irrelevant" samples.


How Does the Check Work?
------------------------
Ideally we would like to directly find images which are outliers, but this is computationally expensive and does not
have a clear and explainable results. Therefore, we use image properties in order to find outliers (such as brightness,
aspect ratio etc.) which are much more efficient to compute, and each outlier is easily explained.

We use `Interquartile Range <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to define our upper
and lower limit for the properties' values.


Which Image Properties Are Used?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default the checks use the built-in image properties, and it's also possible to replace the default properties
with custom ones. For the list of the built-in image properties and explanation about custom properties refer to
:doc:`vision properties </user-guide/vision/vision_properties>`.

"""

#%%
# Run the Check
# -------------
# For the example we will load COCO object detection data, and will run the check with the default properties.
#
# .. note::
#   In this example, we use the pytorch version of the coco dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.detection.coco_tensorflow import load_dataset

from deepchecks.vision.checks import ImagePropertyOutliers
from deepchecks.vision.datasets.detection.coco_torch import load_dataset

train_data = load_dataset(train=True, object_type='VisionData')
check = ImagePropertyOutliers()
result = check.run(train_data)
result

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# Observe Graphic Result
# ^^^^^^^^^^^^^^^^^^^^^^
# The check shows a section for each property. In each section we show the number of outliers and the non-outlier
# property range, and also the images with the lowest and highest values for the property.
#
# For example in property "RMS Contrast" we can see that only 3 outliers were found, 1 below the normal property range
# and 2 above. Now we can inspect these images and decide if we wish to ignore these kinds of samples or if we would
# like the model to be able to support them, in which case we may take a close look into the model's predictions on
# these samples.
#
# Observe Result Value
# ^^^^^^^^^^^^^^^^^^^^
# The check returns CheckResult object with a property 'value' on it which contain the information that was calculated
# in the check's run.

result.value
