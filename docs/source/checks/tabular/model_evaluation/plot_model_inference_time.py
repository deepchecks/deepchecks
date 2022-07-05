# -*- coding: utf-8 -*-
"""
.. _plot_tabular_model_inference_time:

Model Inference Time
********************
"""

#%%

from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ModelInferenceTime

#%%

iris = load_iris(as_frame=True)
train, test = train_test_split(iris.frame, test_size=0.33, random_state=42)

train_ds = Dataset(train, features=iris.feature_names, label='target')
test_ds = Dataset(test, features=iris.feature_names, label='target')

clf = AdaBoostClassifier().fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])

#%%

check = ModelInferenceTime()
result = check.run(test_ds, clf)

#%%
# Instantiating check instance with condition
# ===========================================

check = ModelInferenceTime().add_condition_inference_time_less_than(0.00001)
result = check.run(test_ds, clf)
