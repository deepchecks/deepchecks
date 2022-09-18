# -*- coding: utf-8 -*-
"""
.. _plot_tabular_model_inference_time:

Model Inference Time
*********************
This notebook provides an overview for using and understanding the Model Inference Time check.

**Structure:**

* `What is the Model Inference Time check? <#what-is-the-model-inference-time-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__


What is the Model Inference Time check?
========================================
The ``ModelInferenceTime`` check measures the model's average inference time (in seconds) per sample.
Inference time is an important metric for prediction models, especially in real-time applications and
data streaming processes, where a fast runtime can affect the user's experience or the overall system
load.

"""

#%%
# Imports
# =============
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ModelInferenceTime

#%%
# Generate data & model
# =======================
iris = load_iris(as_frame=True)
train, test = train_test_split(iris.frame, test_size=0.33, random_state=42)

train_ds = Dataset(train, features=iris.feature_names, label='target')
test_ds = Dataset(test, features=iris.feature_names, label='target')

clf = AdaBoostClassifier().fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])

#%%
# Run the check
# =============
check = ModelInferenceTime()
check.run(test_ds, clf)

#%%
# Define a condition
# ===================
# A condition for the average inference time per sample can be defined.
# Here, we define the threshold to be 0.00001 seconds.
check = ModelInferenceTime().add_condition_inference_time_less_than(value=0.00001)
check.run(test_ds, clf)
