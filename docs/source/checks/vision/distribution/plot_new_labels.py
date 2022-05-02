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
do not apper in the training set.
"""

# %%
# Imports
# =======
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.checks import NewLabels


# %%
# Run the Check
# =============

coco_train = coco.load_dataset(train=True, object_type='VisionData', shuffle=False)
coco_test = coco.load_dataset(train=False, object_type='VisionData', shuffle=False)
check = NewLabels().add_condition_new_label_percentage_not_greater_than(0.05)
check.run(coco_train, coco_test)
