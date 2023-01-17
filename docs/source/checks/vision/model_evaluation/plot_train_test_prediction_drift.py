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

from deepchecks.vision.checks import TrainTestPredictionDrift
from deepchecks.vision.datasets.classification.mnist_torch import (load_dataset,
                                                                   load_model)

#%%
# Loading data and model:
# -----------------------


train_ds = load_dataset(train=True, batch_size=64, object_type='VisionData')
test_ds = load_dataset(train=False, batch_size=64, object_type='VisionData')

#%%

model = load_model()

#%%
# Running TrainTestLabelDrift on classification
# ---------------------------------------------

check = TrainTestPredictionDrift()
result = check.run(train_ds, test_ds, model)
result

#%%
# If you have a GPU, you can speed up this check by passing it as an argument to .run() as device=<your GPU>
#
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# Understanding the results
# -------------------------
# We can see there is almost no drift between the train & test labels. This means the
# split to train and test was good (as it is balanced and random). Let's check the
# performance of a simple model trained on MNIST.

from deepchecks.vision.checks import ClassPerformance

ClassPerformance().run(train_ds, test_ds, model)

#%%
# MNIST with label drift
# ======================
# Now, let's try to separate the MNIST dataset in a different manner that will result
# in a prediction drift, and see how it affects the performance. We are going to create
# a custom collate_fn in the test dataset, that will select samples with class 0
# in a 1/10 chances.

import torch

mnist_dataloader_train = load_dataset(train=True, batch_size=64, object_type='DataLoader')
mnist_dataloader_test = load_dataset(train=False, batch_size=64, object_type='DataLoader')
full_mnist = torch.utils.data.ConcatDataset([mnist_dataloader_train.dataset, mnist_dataloader_test.dataset])

#%%

train_dataset, test_dataset = torch.utils.data.random_split(full_mnist, [60000,10000], generator=torch.Generator().manual_seed(42))

#%%
# Inserting drift to the test set
# -------------------------------

import numpy as np
from torch.utils.data._utils.collate import default_collate

np.random.seed(42)

images, labels = next(iter(mnist_dataloader_test))


def collate_test(batch):
    modified_batch = []
    for item in batch:
        image, label = item
        if label == 0:
            if np.random.randint(5) == 0:
                modified_batch.append(item)
            else:
                modified_batch.append((images[0], 1))
        else:
            modified_batch.append(item)

    return default_collate(modified_batch)

mod_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
mod_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, collate_fn=collate_test)

#%%
from deepchecks.vision.datasets.classification.mnist_torch import MNISTData

mod_train_ds = MNISTData(mod_train_loader)
mod_test_ds = MNISTData(mod_test_loader)

#%%
# Run the check
# -------------

check = TrainTestPredictionDrift()
result = check.run(mod_train_ds, mod_test_ds, model)
result

#%%
# Add a condition
# ---------------
# We could also add a condition to the check to alert us to changes in the prediction
# distribution, such as the one that occurred here.

check = TrainTestPredictionDrift().add_condition_drift_score_less_than()
result = check.run(mod_train_ds, mod_test_ds, model)
result

#%%
# As we can see, the condition alerts us to the present of drift in the prediction.

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

result = ClassPerformance().run(mod_train_ds, mod_test_ds, model)
result

#%%
# Inferring the results
# ---------------------

# We can see the drop in the precision of class 0, which was caused by the class
# imbalance indicated earlier by the label drift check.

#%%
# Run the Check on an Object Detection Task (COCO)
# ================================================

from deepchecks.vision.datasets.detection.coco import load_dataset, load_model

train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')
model = load_model(pretrained=True)

#%%

check = TrainTestPredictionDrift()
result = check.run(train_ds, test_ds, model)
result

#%%
# Prediction drift is detected!
# -----------------------------
# We can see that the COCO128 contains a drift in the out of the box dataset. In
# addition to the prediction count per class, the prediction drift check for object
# detection tasks include drift calculation on certain measurements, like the
# bounding box area and the number of bboxes per image.
