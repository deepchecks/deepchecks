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

What Is the Purpose of the Check?
=================================
The Mean Average Precision Report evaluates the
`mAP metric <https://manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html>`__
on the given model and data, plots the AP on graph, and returns the mAP values per bounding box size category
(small, medium, large).
This check only works on the Object Detection task.

"""

# %%
# Generate Data and Model
# =================================
# We generate a sample dataset of 128 images from the `COCO dataset <https://cocodataset.org/#home>`__,
# and using the `YOLOv5 model <https://github.com/ultralytics/yolov5>`__.
#


from deepchecks.vision.checks import MeanAveragePrecisionReport
from deepchecks.vision.datasets.detection import coco

yolo = coco.load_model(pretrained=True)
test_ds = coco.load_dataset(train=False, object_type='VisionData')

# %%
# Run the check
# =================================

check = MeanAveragePrecisionReport()
result = check.run(test_ds, yolo)
result

# %%
# Observe the check’s output
# --------------------------
# The result value is a dataframe that has the Mean Average Precision score for different bounding box area sizes.
# We report the mAP for different IoU thresholds: 0.5, 0.75 and an average of mAP values
# for IoU thresholds between 0.5 and 0.9 (with a jump size of 0.05).

result.value

#%%
# Define a condition 
# ==================
# We can define a condition that checks whether our model's mean average precision score is not less than
# a given threshold for all bounding box sizes.

check = MeanAveragePrecisionReport().add_condition_average_mean_average_precision_not_less_than(0.4)
result = check.run(test_ds, yolo)
result.show(show_additional_outputs=False)
