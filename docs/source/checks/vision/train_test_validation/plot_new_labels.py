# -*- coding: utf-8 -*-
"""
.. _plot_vision_new_labels:

New Labels
==========

This notebooks provides an overview for using and understanding the New Labels check.

**Structure:**

* `How the check works <#How-the-check-works>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__


How the check works
-------------------
In this check we count the frequency of each class id in the test set then check which of them
do not appear in the training set. Note that by default this check run on a sample of the data set and so it is
possible that class ids that are rare in the train set will also be considered as new labels in the test set.
"""

# %%
# Run the Check
# -------------
#
# .. note::
#   In this example, we use the pytorch version of the coco dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.detection import coco_tensorflow as coco

from deepchecks.vision.datasets.detection import coco_torch as coco
from deepchecks.vision.checks import NewLabels

coco_train = coco.load_dataset(train=True, object_type='VisionData', shuffle=False)
coco_test = coco.load_dataset(train=False, object_type='VisionData', shuffle=False)

result = NewLabels().run(coco_train, coco_test)
result

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.

# %%
# Observe the check’s output
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The check searches for new labels in the test set. The value output is a dictionary containing of appearances of each
# newly found class_id in addition to the total number of labels in the test set for comparison purposes.

result.value

# %%
# Define a condition
# -------------------
# The check has a default condition which can be defined. The condition verifies that the ratio of new labels out of
# the total number of labels in the test set is smaller than a given threshold. If the check is run with the default
# sampling mechanism we recommend on setting the condition threshold to a small percentage instead of setting it
# to 0.
#

check = NewLabels().add_condition_new_label_ratio_less_or_equal(0.05)
check.run(coco_train, coco_test)

# %%
# In this case the condition identified that a major portion of the test set labels do not appear in the training set.
