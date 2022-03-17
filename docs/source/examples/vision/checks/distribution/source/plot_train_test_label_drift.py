# -*- coding: utf-8 -*-
"""
Train Test Label Drift
**********************
This notebooks provides an overview for using and understanding the vision label drift check.

**Structure:**

* `What is a label drift? <#what-is-a-label-drift>`__
* `Run check on a Classification task <#run-the-check-on-a-classification-task-mnist>`__
* `Run check on an Object Detection task <#run-the-check-on-an-object-detection-task-coco>`__

What is a label drift?
======================
The term drift (and all it's derivatives) is used to describe any change in the data
compared to the data the model was trained on. Specifically, label drift indicates
changes in the label we are trying to predict.

Causes of label drift include:

* Natural drift in the data, such as lighting (brightness) changes between summer and winter.
* Labeling issues, such as an analyst drawing incorrect bounding boxes for an object
  detection task.

It may be the case that the change in the distribution of the label is due solely
to a change in the type of samples we're receiving (for example, we're now getting
more images of bright days) or it could be due to concept drift, which is a term used
to describe a situation in which the relation between the data and the target has
changed (for example, Covid has caused less people to be outside even in bright images).
The latter type of drift between the training set and the test set will likely make the
model to be prone to errors because the function that the model is trying to approximate
has changed, and the first may also lead to a change in the performance due to different
populations of samples being present in the data or due to trying to predict on outlier
samples for which the model was not fitted. For both cases, checking for drift in the
label variable between the train and test datasets may serve to notify us that such a
phenomenon is occurring.

How deepchecks detects label drift
----------------------------------
There are many methods to detect label drift, that usually include statistical methods
that aim to measure difference between distribution of 2 given label sets. We
experimented with various approaches and found that for detecting drift between 2
one-dimensional distribution, the following 2 methods give the best results:

* `Population Stability Index (PSI) <https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf>`__
* `Wasserstein metric (Earth Movers Distance) <https://en.wikipedia.org/wiki/Wasserstein_metric>`__

For numerical features, the check uses the Earth Movers Distance method and for the
categorical features it uses the PSI. The check calculates drift between train dataset
and test dataset per feature, using these 2 statistical measures.

Different measurement on label
==============================
In computer vision specifically, our labels may be complex, and measuring their drift
is not a straightforward task. Therefore, we calculate drift on different measures on
labels. For now, in deepchecks, we support these measurements (on which we calculate
drift):

* For both classification and object detection tasks, we calculate drift in the
  distribution of classes.
* For object detection tasks, we also calculate drift in the distribution of bounding
  box areas and distribution of number of bounding boxes per image.

Run the check on a Classification task (MNIST)
==============================================
Imports
-------
"""

#%%
from deepchecks.vision.checks import TrainTestLabelDrift

#%%
# Loading Data
# ------------

from deepchecks.vision.datasets.classification.mnist import load_dataset

train_ds = load_dataset(train=True, batch_size=64, object_type='VisionData')
test_ds = load_dataset(train=False, batch_size=1000, object_type='VisionData')

#%%
# Running TrainTestLabelDrift on classification
# ---------------------------------------------

check = TrainTestLabelDrift()
check.run(train_ds, test_ds)

#%%
# Understanding the results
# -------------------------
# We can see there is almost no drift between the train & test labels. This means the
# split to train and test was good (as it is balanced and random). Let's check the
# performance of a simple model trained on MNIST.

from deepchecks.vision.datasets.classification.mnist import load_model as load_mnist_model
from deepchecks.vision.checks import ClassPerformance

mnist_model = load_mnist_model(pretrained=True)
ClassPerformance().run(train_ds, test_ds, mnist_model)

#%%
# MNIST with label drift
# ======================
# Now, let's try to separate the MNIST dataset in a different manner that will result
# in a label drift, and see how it affects the performance. We are going to create a
# custom `collate_fn`` in the test dataset, that will select samples with class 0 in
# a 1/10 chances.

import torch

mnist_dataloader_train = load_dataset(train=True, batch_size=64, object_type='DataLoader')
mnist_dataloader_test = load_dataset(train=False, batch_size=1000, object_type='DataLoader')
full_mnist = torch.utils.data.ConcatDataset([mnist_dataloader_train.dataset, mnist_dataloader_test.dataset])

#%%
train_dataset, test_dataset = torch.utils.data.random_split(full_mnist, [60000,10000], generator=torch.Generator().manual_seed(42))

#%%
# Inserting drift to the test set
# -------------------------------

from torch.utils.data._utils.collate import default_collate
import numpy as np

np.random.seed(42)

def collate_test(batch):
    modified_batch = []
    for item in batch:
        image, label = item
        if label == 0:
            if np.random.randint(10) == 0:
                modified_batch.append(item)
        else:
            modified_batch.append(item)
            
    return default_collate(modified_batch)

mod_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
mod_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, collate_fn=collate_test)

#%%

from deepchecks.vision.datasets.classification.mnist import MNISTData

mod_train_ds = MNISTData(mod_train_loader)
mod_test_ds = MNISTData(mod_test_loader)

#%%
# Run the check
# =============

check = TrainTestLabelDrift()
check.run(mod_train_ds, mod_test_ds)

#%%
# Add a condition
# ---------------
# We could also add a condition to the check to alert us to changes in the label
# distribution, such as the one that occurred here.

check = TrainTestLabelDrift().add_condition_drift_score_not_greater_than()
check.run(mod_train_ds, mod_test_ds)

# As we can see, the condition alerts us to the present of drift in the label.

#%%
# Results
# -------
# We can see the check successfully detects the (expected) drift in class 0
# distribution between the train and test sets

#%%
# But how does this affect the performance of the model?
# ------------------------------------------------------

ClassPerformance().run(mod_train_ds, mod_test_ds, mnist_model)

#%%
# Inferring the results
# ---------------------
# We can see the drop in the precision of class 0, which was caused by the class imbalance indicated earlier by the label drift check.

#%%
# Run the check on an Object Detection task (COCO)
# ================================================

from deepchecks.vision.datasets.detection.coco import load_dataset

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
