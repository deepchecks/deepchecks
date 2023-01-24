# -*- coding: utf-8 -*-
"""
.. _plot_vision_train_test_prediction_drift:

Train Test Prediction Drift
***************************

This notebooks provides an overview for using and understanding the vision prediction drift check.

**Structure:**

* `What Is Prediction Drift? <#what-is-prediction-drift>`__
* `Which Prediction Properties Are Used? <#which-prediction-properties-are-used>`__
* `Run Check on a Classification Task <#run-the-check-on-a-classification-task-mnist>`__
* `Run Check on an Object Detection Task <#run-the-check-on-an-object-detection-task-coco>`__

What Is Prediction Drift?
=========================
Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Prediction drift is when drift occurs in the prediction itself.
Calculating prediction drift is especially useful in cases
in which labels are not available for the test dataset, and so a drift in the predictions
is a direct indication that a change that happened in the data has affected the model's
predictions. If labels are available, it's also recommended to run the
:doc:`Label Drift check </checks_gallery/vision/train_test_validation/plot_train_test_label_drift>`.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide>`

How Deepchecks Detects Prediction Drift
---------------------------------------

This check detects prediction drift by using :ref:`univariate measures <drift_detection_by_univariate_measure>`
on the prediction properties.

Using Prediction Properties to Detect Prediction Drift
------------------------------------------------------

In computer vision specifically, our predictions may be complex, and measuring their drift
is not a straightforward task. Therefore, we calculate drift on different :doc:`properties of the prediction</user-guide/vision/vision_properties>`,
on which we can directly measure drift.

Which Prediction Properties Are Used?
=====================================

================  ===================================  ==========
Task Type         Property name                        What is it
================  ===================================  ==========
Classification    Samples Per Class                    Number of images per class
Object Detection  Samples Per Class                    Number of bounding boxes per class
Object Detection  Bounding Box Area                    Area of bounding box (height * width)
Object Detection  Number of Bounding Boxes Per Image   Number of bounding box objects in each image
================  ===================================  ==========


Run the Check on a Classification Task (MNIST)
==============================================
"""

#%%
# Imports
# -------
#
# .. note::
#   In this example, we use the pytorch version of the mnist dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.classification.mnist_tensorflow import load_dataset

from deepchecks.vision.checks import TrainTestPredictionDrift
from deepchecks.vision.datasets.classification.mnist_torch import load_dataset

#%%
# Load Dataset
# ----------------


train_ds = load_dataset(train=True, batch_size=64, object_type='VisionData')
test_ds = load_dataset(train=False, batch_size=64, object_type='VisionData')


#%%
# Running TrainTestPredictionDrift on classification
# ---------------------------------------------

check = TrainTestPredictionDrift()
result = check.run(train_ds, test_ds)
result

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# Understanding the results
# -------------------------
# We can see there is almost no drift between the train & test predictions. This means the
# split to train and test was good (as it is balanced and random). Let's check the
# performance of a simple model trained on MNIST.

from deepchecks.vision.checks import ClassPerformance

ClassPerformance().run(train_ds, test_ds)

#%%
# MNIST with prediction drift
# ======================
# Now, let's try to separate the MNIST dataset in a different manner that will result
# in a prediction drift, and see how it affects the performance. We are going to create a
# custom `collate_fn`` in the test dataset, that will select a few of the samples with class 0
# and change their most of their predicted classes to 1.

#%%
# Inserting drift to the test set
# -------------------------------

import numpy as np
import torch

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
                    # In 9/10 cases, the prediction vector will change to match the label
                    if np.random.randint(10) != 0:
                        batch_dict['predictions'][i] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

        return batch_dict
    return collate_fn_with_label_drift


mod_test_ds = load_dataset(train=False, batch_size=1000, object_type='VisionData')
mod_test_ds._batch_loader.collate_fn = generate_collate_fn_with_label_drift(mod_test_ds._batch_loader.collate_fn)

#%%
# Run the check
# -------------

check = TrainTestPredictionDrift()
result = check.run(train_ds, mod_test_ds)
result

#%%
# Add a condition
# ---------------
# We could also add a condition to the check to alert us about changes in the prediction
# distribution, such as the one that occurred here.

check = TrainTestPredictionDrift().add_condition_drift_score_less_than()
result = check.run(train_ds, mod_test_ds)
result

#%%
# As we can see, the condition alerts us to the presence of drift in the predictions.

#%%
# Results
# -------
# We can see the check successfully detects the (expected) drift in class 0 distribution
# between the train and test sets. It means the the model correctly predicted 0 for
# those samples and so we're seeing drift in the predictions as well as the labels.
# We note that this check enabled us to detect the presence of label drift (in this case)
# without needing actual labels for the test data.

#%%
# But how does this affect the performance of the model?
# ------------------------------------------------------

result = ClassPerformance().run(train_ds, mod_test_ds)
result

#%%
# Inferring the results
# ---------------------
# We can see the drop in the precision of class 0, which was caused by the class
# imbalance indicated earlier by the label drift check.

#%%
# Run the Check on an Object Detection Task (COCO)
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

check = TrainTestPredictionDrift()
result = check.run(train_ds, test_ds)
result

#%%
# Prediction drift is detected!
# -----------------------------
# We can see that the COCO128 contains a drift in the out of the box dataset. In
# addition to the prediction count per class, the prediction drift check for object
# detection tasks include drift calculation on certain measurements, like the
# bounding box area and the number of bboxes per image.
