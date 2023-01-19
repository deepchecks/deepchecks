# -*- coding: utf-8 -*-
"""
.. _plot_vision_mean_average_recall_report:

Mean Average Recall Report
**************************

This notebooks provides an overview for using and understanding the mean average recall report check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate Dataset <#generate-dataset>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================
The Mean Average Recall Report evaluates the
`mAR metric <https://manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html>`__
on the given model and data, and returns the mAR values per bounding box size category (small, medium, large).
This check only works on the Object Detection task.
"""

#%%
# Imports
# -------
#
# .. note::
#   In this example, we use the pytorch version of the coco dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.detection import coco_tensorflow as coco

from deepchecks.vision.checks import MeanAverageRecallReport
from deepchecks.vision.datasets.detection import coco_torch as coco

#%%
# Generate Dataset
# ----------------
# We generate a sample dataset of 128 images from the `COCO dataset <https://cocodataset.org/#home>`__,
# and using the `YOLOv5 model <https://github.com/ultralytics/yolov5>`__.
#
# For the label formatter - our dataset returns exactly the accepted format, so our
# formatting function is the simple `lambda x: x` function.

test_ds = coco.load_dataset(train=False, object_type='VisionData')

#%%
# Run the check
# -------------

check = MeanAverageRecallReport()
result = check.run(test_ds)
result

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# Observe the check’s output
# --------------------------
# The result value is a dataframe that has the average recall score per each area range and IoU.

result.value

#%%
# Define a condition 
# ==================
# We can define a condition that checks whether our model's average recall score is not less than
# a given threshold

check = MeanAverageRecallReport().add_condition_test_average_recall_greater_than(0.4)
result = check.run(test_ds)
result.show(show_additional_outputs=False)
