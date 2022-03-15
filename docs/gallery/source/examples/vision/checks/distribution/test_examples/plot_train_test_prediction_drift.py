# -*- coding: utf-8 -*-
"""
Train Test Prediction Drift
***************************
This notebooks provides an overview for using and understanding the vision prediction drift check.

**Structure:**

* `What is a prediction drift? <#what-is-a-prediction-drift>`__
* `Run check on a Classification task <#run-the-check-on-a-classification-task-mnist>`__
* `Run check on an Object Detection task <#run-the-check-on-an-object-detection-task-coco>`__

What is a prediction drift?
===========================
The term drift (and all it's derivatives) is used to describe any change in the data compared
to the data the model was trained on. Prediction drift refers to the case in which a change
in the data (data/feature drift) has happened and as a result, the distribution of the
models' prediction has changed. Calculating prediction drift is especially useful in cases
in which labels are not available for the test dataset, and so a drift in the predictions
is out only indication that a changed has happened in the data that actually affects model
predictions. If labels are available, it's also recommended to run the `Label Drift Check
</examples/vision/checks/distribution/test_autoexamples/plot_train_test_label_drift.html>`__.

There are two main causes for prediction drift:

* Change in the sample population. In this case, the underline phenomenon we're trying
  to predict behaves the same, but we're not getting the same types of samples. An example
  for that is seasonal changes that cause a changes in the brightness of the images we're
  receiving for inference. In extreme, this kind of change may take us into domains that
  the model was not trained to function in and may degrade model performance. Even if
  that is not the case, just changing the population can result in a prediction drift if
  the different population receive different model predictions.
* Concept drift, which is a case in which the underline relation between the data and 
  the label has changed. That case, also of greater interest to us, won't necessarily
  result in prediction drift but it may be accompanied with a multivariate change in
  the data distribution which will affect model predictions.

In the context of machine learning, drift between the training set and the test set will
likely make the model to be prone to errors. In other words, this means that the model
was trained on data that is different from the current test data, thus it will probably
make more mistakes predicting the target variable.

How deepchecks detects prediction drift
---------------------------------------
There are many methods to detect drift between two distributions, that usually include
statistical methods that aim to measure difference between distribution of 2 given
sets of predictions. We experimented with various approaches and found that for
detecting drift between 2 one-dimensional distribution, the following 2 methods give
the best results:

* `Population Stability Index (PSI) <https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf>`__
* `Wasserstein metric (Earth Movers Distance) <https://en.wikipedia.org/wiki/Wasserstein_metric>`__

For numerical features, the check uses the Earth Movers Distance method and for the categorical
features it uses the PSI. The check calculates drift between train dataset and test dataset
per feature, using these 2 statistical measures.

Different measurement on predictions
====================================
In computer vision specifically, our predictions may be complex, and measuring their
drift is not a straightforward task (for example, object detection predictions are
bounding boxes). Therefore, we calculate drift on different measures on labels. For now,
in deepchecks, we support these measurements (on which we calculate drift):

* For both classification and object detection tasks, we calculate drift in the
  distribution of classes.
* For object detection tasks, we also calculate drift in the distribution of bounding box
  areas and distribution of number of bounding boxes per image.

Run the check on a Classification task (MNIST)
==============================================
"""

#%%
# Imports
# -------

from deepchecks.vision.checks import TrainTestPredictionDrift

#%%
# Loading data and model:
# -----------------------

from deepchecks.vision.datasets.classification.mnist import load_dataset, load_model

train_ds = load_dataset(train=True, batch_size=64, object_type='VisionData')
test_ds = load_dataset(train=False, batch_size=64, object_type='VisionData')

#%%

model = load_model()

#%%
# Running TrainTestLabelDrift on classification
# ---------------------------------------------

check = TrainTestPredictionDrift()
check.run(train_ds, test_ds, model)

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

# Run the check
# -------------

check = TrainTestPredictionDrift()
check.run(mod_train_ds, mod_test_ds, model)

# Add a condition
# ---------------
# We could also add a condition to the check to alert us to changes in the prediction
# distribution, such as the one that occurred here.

check = TrainTestPredictionDrift().add_condition_drift_score_not_greater_than()
check.run(mod_train_ds, mod_test_ds, model)

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

ClassPerformance().run(mod_train_ds, mod_test_ds, model)

#%%
# Inferring the results
# ---------------------

# We can see the drop in the precision of class 0, which was caused by the class
# imbalance indicated earlier by the label drift check.

#%%
# Run the check on an Object Detection task (COCO)
# ================================================

from deepchecks.vision.datasets.detection.coco import load_dataset, load_model

train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')
model = load_model(pretrained=True)

#%%

check = TrainTestPredictionDrift()
check.run(train_ds, test_ds, model)

#%%
# Prediction drift is detected!
# -----------------------------
# We can see that the COCO128 contains a drift in the out of the box dataset. In
# addition to the prediction count per class, the prediction drift check for object
# detection tasks include drift calculation on certain measurements, like the
# bounding box area and the number of bboxes per image.
