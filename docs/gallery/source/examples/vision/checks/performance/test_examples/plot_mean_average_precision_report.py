# -*- coding: utf-8 -*-
"""
Mean Average Precision Report
*****************************
This notebooks provides an overview for using and understanding the mean average precision report check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data & model <#generate-data-and-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================
The mean average precision report evaluates the mean average precision metric on the
given model and data. The check only works on object detection as it computes the
value with the IoU and Area size.
"""

#%%
# Imports
# -------

import numpy as np

from deepchecks.vision.base import VisionData
from deepchecks.vision.checks.performance import MeanAveragePrecisionReport
from deepchecks.vision.datasets.detection import coco

#%%
# Generate Data and Model
# -----------------------
# We generate a sample dataset of 128 images from the `COCO dataset <https://cocodataset.org/#home>`__,
# and using the `YOLOv5 model <https://github.com/ultralytics/yolov5>`__.
#
# For the label formatter - our dataset returns exactly the accepted format, so our
# formatting function is the simple `lambda x: x` function.

yolo = coco.load_model(pretrained=True)

test_ds = coco.load_dataset(train=False, object_type='VisionData')

#%%
# Run the check
# -------------

check = MeanAveragePrecisionReport()
result = check.run(test_ds, yolo)
result

#%%
# Observe the check’s output
# --------------------------
# The result value is a dataframe that has the average precision score per each area range and IoU.

result.value

#%%
# Define a condition 
# ==================
# We can define a condition that enforce our model's average precision score is not less than
# a given threshold

check = MeanAveragePrecisionReport().add_condition_test_average_precision_not_less_than(0.4)
result = check.run(test_ds, yolo)
result.show(show_additional_outputs=False)
