# -*- coding: utf-8 -*-
"""
Model Error Analysis check
==========================

This notebooks provides an overview for using and understanding the
model error analysis check.

**Structure:**

-  `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
-  `Classification <#classification-performance-report>`__

   -  `Generate data & model <#generate_c>`__
   -  `Run the check <#run_check_c>`__

-  `Object Detection <#object-detection-class-performance>`__

   -  `Generate data & model <#generate_o>`__
   -  `Run the check <#run_check_o>`__

What is the purpose of the check?
---------------------------------
TODO
"""

#%%
# Imports
# -------

from deepchecks.vision.checks.performance import ModelErrorAnalysis

#%%
# Classification Performance Report
# ---------------------------------
#
# .. _generate_c:
#
# Generate data and model:
# ~~~~~~~~~~~~~~~~~~~~~~~~
from deepchecks.vision.datasets.classification import mnist

mnist_model = mnist.load_model()
train_ds = mnist.load_dataset(train=True, object_type='VisionData')
test_ds = mnist.load_dataset(train=False, object_type='VisionData')

#%%
# .. _run_check_c:
#
# Run the check:
# ~~~~~~~~~~~~~~~~
check = ModelErrorAnalysis(min_error_model_score=0)
check.run(train_ds, test_ds, mnist_model)

#%%
# Object Detection Class Performance
# ----------------------------------
#
# For object detection tasks - the default metric that is being calculated
# it the Average Precision. The definition of the Average Precision is
# identical to how the COCO dataset defined it - mean of the average
# precision per class, over the range [0.5, 0.95, 0.05] of IoU thresholds.

import numpy as np
from deepchecks.vision.datasets.detection import coco

#%%
# .. _generate_o:
#
# Generate Data and Model
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# We generate a sample dataset of 128 images from the `COCO
# dataset <https://cocodataset.org/#home>`__, and using the `YOLOv5
# model <https://github.com/ultralytics/yolov5>`__

yolo = coco.load_model(pretrained=True)

train_ds = coco.load_dataset(train=True, object_type='VisionData')
test_ds = coco.load_dataset(train=False, object_type='VisionData')

#%%
# .. _run_check_o:

# Run the check:
# ~~~~~~~~~~~~~~~~

check = ModelErrorAnalysis(min_error_model_score=-1)
check.run(train_ds, test_ds, yolo)
