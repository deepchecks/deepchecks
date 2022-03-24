# -*- coding: utf-8 -*-
"""
Image Property Outliers
=======================
This notebooks provides an overview for using and understanding the image property
outliers check, used to detect outliers in simple image properties in a dataset.

**Structure:**

* `Why Check For Outliers? <#why-check-for-outliers>`__
* `How Does the Check Work? <#how-does-the-check-work>`__
* `Which Image Properties Are Used? <#which-image-properties-are-used>`__
* `Run The Check <#run-the-check>`__


Why Check For Outliers?
-----------------------
Examining outliers may help you gain insights that you couldn't have reached from taking an aggregate look or by
inspecting random samples.. For example, it may help you understand you have some dirty samples (e.g.
an image that is completely black), or samples you didn't expect to have (sample with extreme aspect ratio).
In some cases, these outliers may help to debug some performance discrepancies (the model can be excused for failing on
a totally dark image). In more extreme cases, the outlier samples may indicate the presence of samples interfering with
the model's training by teaching the model to fit "irrelevant" samples.


How Does the Check Work?
------------------------
Ideally we would like to directly find images which are outliers. But since this is more complex and not
straight-forward to achieve we search for outliers on the image's properties (such as brightness, aspect ratio etc.)
and search for outliers in a single-dimension data.

There are many methods to detect outliers on a single dimension data. In this check we are using
`Interquartile Range <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to define our upper and lower
limit for the properties' values.


Which Image Properties Are Used?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default the checks use the built-in image properties, and it's also possible to replace the default properties
with custom ones. For the list of the built-in image properties and explanation about custom properties refer to
:doc:`vision properties </user-guide/vision/visio_properties>`.

"""

#%%
# Run the check
# -------------
# For the example we will load COCO object detection data, and will run the check with the default properties.

from deepchecks.vision.datasets.detection.coco import load_dataset
from deepchecks.vision.checks import ImagePropertyOutliers

train_data = load_dataset(train=True, object_type='VisionData')
check = ImagePropertyOutliers()
result = check.run(train_data)
result

#%%
# Observe Result Value
# ^^^^^^^^^^^^^^^^^^^^
# The check returns CheckResult object with a property 'value' on it which contain the information that was calculated
# in the check's run.

result.value


