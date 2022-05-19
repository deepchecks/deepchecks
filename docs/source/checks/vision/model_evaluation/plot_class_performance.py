# -*- coding: utf-8 -*-
"""
Class Performance
***********************

This notebooks provides an overview for using and understanding the class performance check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Classification <#classification-performance-report>`__

  - `Generate data & model <#generate-data-and-model>`__
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

In addition to the default metrics, the check supports custom metrics that
should be implemented using the `torch.ignite.Metric
<https://pytorch.org/ignite/metrics.html#how-to-create-a-custom-metric>`__ API.
These can be passed as a list using the alternative_metrics parameter of the
check, which will override the default metrics.

"""

#%%
# Imports
# -------

from deepchecks.vision.checks import ClassPerformance
from deepchecks.vision.datasets.classification import mnist

#%%
# Classification Performance Report
# =================================
# Generate data and model:
# ------------------------


mnist_model = mnist.load_model()
train_ds = mnist.load_dataset(train=True, object_type='VisionData')
test_ds = mnist.load_dataset(train=False, object_type='VisionData')

#%%
# Run the check
# -------------

check = ClassPerformance()
check.run(train_ds, test_ds, mnist_model)

#%%
# Object Detection Class Performance
# ==================================
# For object detection tasks - the default metric that is being calculated it the
# Average Precision. The definition of the Average Precision is identical to how
# the COCO dataset defined it - mean of the average precision per class, over
# the range [0.5, 0.95, 0.05] of IoU thresholds.

from deepchecks.vision.datasets.detection import coco

#%%
# Generate Data and Model
# -----------------------
# We generate a sample dataset of 128 images from the `COCO dataset <https://cocodataset.org/#home>`__,
# and using the `YOLOv5 model <https://github.com/ultralytics/yolov5>`__.

yolo = coco.load_model(pretrained=True)

train_ds = coco.load_dataset(train=True, object_type='VisionData')
test_ds = coco.load_dataset(train=False, object_type='VisionData')

#%%
# Run the check
# -------------

check = ClassPerformance(show_only='best')
check.run(train_ds, test_ds, yolo)

#%%
# Define a Condition
# ==================
# We can also define a condition to validate that our model performance is above a certain threshold.
# The condition is defined as a function that takes the results of the check as input and
# returns a ConditionResult object.

check = ClassPerformance(show_only='worst')
check.add_condition_test_performance_not_less_than(0.2)
result = check.run(train_ds, test_ds, yolo)
result

#%%
# We detected that for several classes our model performance is below the threshold.
