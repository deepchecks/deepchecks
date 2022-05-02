# -*- coding: utf-8 -*-
"""
New Labels
***************

This notebooks provides an overview for using and understanding the New Labels check.

**Structure:**

* `How the check works <#How-the-check-works>`__
* `Run the check <#run-the-check>`__

How the check works
========================
In this check we count the frequency of each class id in the test set then check which of them
do not apper in the training set. By default, checks run on a sample of the data which can create a false appearance of
new labels in the test set. In order to overcome this issue we recommend on setting the condition threshold to a small
percentage instead of setting it to 0. For Object detection tasks, the output images present only bounding boxes with
the relevant class_id.
"""

# %%
# Run the Check
# =============
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.checks import NewLabels

coco_train = coco.load_dataset(train=True, object_type='VisionData', shuffle=False)
coco_test = coco.load_dataset(train=False, object_type='VisionData', shuffle=False)
check = NewLabels().add_condition_new_label_ratio_not_greater_than(0.05)
check.run(coco_train, coco_test)
