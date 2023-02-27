# -*- coding: utf-8 -*-
"""
.. _plot_vision_label_property_outliers:

Label Property Outliers
=======================

This notebooks provides an overview for using and understanding the label property
outliers check, used to detect outliers in simple label properties in a dataset.

**Structure:**

* `Why Check for Label Outliers? <#why-check-for-label-outliers>`__
* `How Does the Check Work? <#how-does-the-check-work>`__
* `Which Label Properties Are Used? <#which-label-properties-are-used>`__
* `Run the Check <#run-the-check>`__


Why Check for Label Outliers?
-----------------------------
Examining outliers may help you gain insights that you couldn't have reached from taking an aggregate look or by
inspecting random samples. For example, it may help you understand you have some corrupt samples (e.g.
a bounding box with area 0), or samples you didn't expect to have (e.g. extreme aspect ratio).
In some cases, these outliers may help debug some performance discrepancies (the model can be excused for failing on
a zero size bounding box). In more extreme cases, the outlier samples may indicate the presence of samples interfering
with the model's training by teaching the model to fit "irrelevant" samples.


How Does the Check Work?
------------------------
In order to find outlier labels we use label properties (such as number of bounding boxes, bounding box area, etc.)

We use `Interquartile Range <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to define our upper
and lower limit for the properties' values.


Which Label Properties Are Used?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For object detection we have default built-in label properties. For other tasks you have to define your own custom label
properties.
For the list of the built-in object detection label properties and explanation about custom properties refer to
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

from deepchecks.vision.checks import LabelPropertyOutliers
from deepchecks.vision.datasets.detection.coco_torch import load_dataset

train_data = load_dataset(train=True, object_type='VisionData')
check = LabelPropertyOutliers()
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
# The check displays a section for each property. In each section we show the number of outliers and the non-outlier
# property range, and also the images with the lowest and highest values for the property. In addition, if the property
# returns a value per bounding box, we then show only the relevant bounding box which resulted in the outlier result.
#
# For example in property "Bounding Box Area (in pixels)" we can see that 80 outliers were found. Now we can inspect
# the samples and decide if we wish to ignore these kinds of samples or if we would
# like the model to be able to support them, in which case we may take a close look into the model's predictions on
# these samples.
#
# Observe Result Value
# ^^^^^^^^^^^^^^^^^^^^
# The check returns CheckResult object with a property 'value' on it which contain the information that was calculated
# in the check's run.

result.value


