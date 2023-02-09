# -*- coding: utf-8 -*-
"""
.. _plot_vision_train_test_label_drift:

Train Test Label Drift
**********************

This notebooks provides an overview for using and understanding label drift check.

**Structure:**

* `What Is Label Drift? <#what-is-label-drift>`__
* `Which Label Properties Are Used? <#which-label-properties-are-used>`__
* `Run check on a Classification task <#run-the-check-on-a-classification-task-mnist>`__
* `Run check on an Object Detection task <#run-the-check-on-an-object-detection-task-coco>`__

What Is Label Drift?
========================
Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Label drift is when drift occurs in the label itself.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide>`.

How Deepchecks Detects Label Drift
------------------------------------

This check detects label drift by using :ref:`univariate measures <drift_detection_by_univariate_measure>`
on the label properties.

Using Label Properties to Detect Label Drift
--------------------------------------------
In computer vision specifically, our labels may be complex, and measuring their drift
is not a straightforward task. Therefore, we calculate drift on different :doc:`properties of the label</user-guide/vision/vision_properties>`,
on which we can directly measure drift.

Which Label Properties Are Used?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
================  ===================================  ==========
Task Type         Property name                        What is it
================  ===================================  ==========
Classification    Samples Per Class                    Number of images per class
Object Detection  Samples Per Class                    Number of bounding boxes per class
Object Detection  Bounding Box Area                    Area of bounding box (height * width)
Object Detection  Number of Bounding Boxes Per Image   Number of bounding box objects in each image
================  ===================================  ==========


Run the check on a Classification task (MNIST)
==============================================
Imports
-------
"""

#%%
#
# .. note::
#   In this example, we use the pytorch version of the mnist dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.classification.mnist_tensorflow import load_dataset

from deepchecks.vision.checks import TrainTestLabelDrift
from deepchecks.vision.datasets.classification.mnist_torch import load_dataset

#%%
# Loading Data
# ------------


train_ds = load_dataset(train=True, batch_size=64, object_type='VisionData')
test_ds = load_dataset(train=False, batch_size=1000, object_type='VisionData')

#%%
# Running TrainTestLabelDrift on classification
# ---------------------------------------------

check = TrainTestLabelDrift()
result = check.run(train_ds, test_ds)
result.show()

#%%
# Understanding the results
# -------------------------
# We can see there is almost no drift between the train & test labels. This means the
# split to train and test was good (as it is balanced and random). Let's check the
# performance of a simple model trained on MNIST.

from deepchecks.vision.checks import ClassPerformance

ClassPerformance().run(train_ds, test_ds)

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

#  ClassPerformance().run(train_ds, test_ds, mnist_model).show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# MNIST with label drift
# ======================
# Now, let's try to separate the MNIST dataset in a different manner that will result
# in a label drift, and see how it affects the performance. We are going to create a
# custom `collate_fn`` in the test dataset, that will select samples with class 0 with
# a probability of 1/10.

#%%
# Inserting drift to the test set
# -------------------------------

import numpy as np

np.random.seed(42)


def generate_collate_fn_with_label_drift(collate_fn):
    def collate_fn_with_label_drift(batch):
        batch_dict = collate_fn(batch)
        images = batch_dict['images']
        labels = batch_dict['labels']
        for i in range(len(images)):
            image, label = images[i], labels[i]
            if label == 0:
                if np.random.randint(5) != 0:
                    batch_dict['labels'][i] = 1

        return batch_dict
    return collate_fn_with_label_drift


mod_test_ds = load_dataset(train=False, batch_size=1000, object_type='VisionData')
mod_test_ds._batch_loader.collate_fn = generate_collate_fn_with_label_drift(mod_test_ds._batch_loader.collate_fn)

#%%
# Run the check
# =============

check = TrainTestLabelDrift()
check.run(train_ds, mod_test_ds)

#%%
# Add a condition
# ---------------
# We could also add a condition to the check to alert us to changes in the label
# distribution, such as the one that occurred here.

check = TrainTestLabelDrift().add_condition_drift_score_less_than()
check.run(train_ds, mod_test_ds)

# As we can see, the condition alerts us to the presence of drift in the label.

#%%
# Results
# -------
# We can see the check successfully detects the (expected) drift in class 0
# distribution between the train and test sets

#%%
# But how does this affect the performance of the model?
# ------------------------------------------------------

ClassPerformance().run(train_ds, mod_test_ds)

#%%
# Understanding the results
# -------------------------------
# We can see the drop in the precision of class 0, which was caused by the class imbalance indicated earlier by the label drift check.

#%%
# Run the check on an Object Detection task (COCO)
# ================================================
#
# .. note::
#   In this example, we use the pytorch version of the coco dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.detection.coco_tensorflow import load_dataset

from deepchecks.vision.datasets.detection.coco_torch import load_dataset

train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')

#%%

check = TrainTestLabelDrift()
check.run(train_ds, test_ds)

#%%
# Label drift is detected!
# ------------------------
# We can see that the COCO128 contains a drift in the out of the box dataset. In
# addition to the label count per class, the label drift check for object detection
# tasks include drift calculation on certain measurements, like the bounding box
# area and the number of bboxes per image.
