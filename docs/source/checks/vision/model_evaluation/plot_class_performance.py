# -*- coding: utf-8 -*-
"""
.. _plot_vision_class_performance:

Class Performance
***********************

This notebooks provides an overview for using and understanding the class performance check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Classification <#classification-performance-report>`__

  - `Generate Dataset <#generate-dataset>`__
  - `Run the check <#run-the-check>`__

* `Object Detection <#object-detection-class-performance>`__

  - `Generate data & model <#id1>`__
  - `Run the check <#id2>`__

What Is the Purpose of the Check?
=================================
The class performance check evaluates several metrics on the given model and data
and returns all of the results in a single check.
The check uses the following default metrics:

=================  ====================
Task Type          Property name
=================  ====================
Classification     Precision
Classification     Recall
Object Detection   `Average Precision <https://manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html>`__
Object Detection   `Average Recall <https://manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html>`__
=================  ====================

In addition to the default metrics, the check supports custom metrics, as detailed in the
:doc:`Metrics Guide </user-guide/general/metrics_guide>`.
These can be passed as a list using the scorers parameter of the check, which will override the default metrics.
"""

#%%
# Imports
# -------
#
# .. note::
#   In this example, we use the pytorch version of the mnist dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.classification import mnist_tensorflow as mnist

from deepchecks.vision.checks import ClassPerformance
from deepchecks.vision.datasets.classification import mnist_torch as mnist

#%%
# Classification Performance Report
# =================================
# Generate Dataset
# -----------------


train_ds = mnist.load_dataset(train=True, object_type='VisionData')
test_ds = mnist.load_dataset(train=False, object_type='VisionData')

#%%
# Run the check
# -------------

check = ClassPerformance()
result = check.run(train_ds, test_ds)
result

#%%
# To display the results in an IDE like PyCharm, you can use the following code:
#
# .. code-block:: python
#
#    result.show_in_window()

#%%
# The result will be displayed in a new window.

#%%
# Object Detection Class Performance
# ==================================
# For object detection tasks - the default metric that is being calculated it the
# Average Precision. The definition of the Average Precision is identical to how
# the COCO dataset defined it - mean of the average precision per class, over
# the range [0.5, 0.95, 0.05] of IoU thresholds.
#
# .. note::
#   In this example, we use the pytorch version of the coco dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.detection import coco_tensorflow as coco

from deepchecks.vision.datasets.detection import coco_torch as coco

#%%
# Generate Dataset
# ----------------
# We generate a sample dataset of 128 images from the `COCO dataset <https://cocodataset.org/#home>`__,
# and using the `YOLOv5 model <https://github.com/ultralytics/yolov5>`__.

train_ds = coco.load_dataset(train=True, object_type='VisionData')
test_ds = coco.load_dataset(train=False, object_type='VisionData')

#%%
# Run the check
# -------------

check = ClassPerformance(show_only='best')
result = check.run(train_ds, test_ds)
result.show()

#%%
# If you have a GPU, you can speed up this check by calling:

# check.run(train_ds, test_ds, yolo, device=<your GPU>)

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

# result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# Define a Condition
# ==================
# We can also define a condition to validate that our model performance is above a certain threshold.
# The condition is defined as a function that takes the results of the check as input and
# returns a ConditionResult object.

check = ClassPerformance(show_only='worst')
check.add_condition_test_performance_greater_than(0.2)
result = check.run(train_ds, test_ds)
result.show()


#%%
# We detected that for several classes our model performance is below the threshold.
