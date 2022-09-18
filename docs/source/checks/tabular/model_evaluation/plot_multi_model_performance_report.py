# -*- coding: utf-8 -*-
"""
.. _plot_tabular_multi_model_performance_report:

Multi Model Performance Report
********************************
This notebook provides an overview for using and understanding the Multi Model Performance Report check.

**Structure:**

* `What is the Multi Model Performance Report check? <#what-is-the-multi-model-performance-report-check>`__
* `Multiclass check <#multiclass-check>`__
* `Regression check <#regression-check>`__


What is the Multi Model Performance Report?
============================================
The ``MultiModelPerformanceReport`` check produces a summary of performance scores for multiple models
on test datasets. The default scorers that are used are F1, Precision and Recall for Classification
and Negative RMSE (Root Mean Square Error), Negative MAE (Mean Absolute Error) and R2 for Regression.
"""

#%%
# Multiclass check
# ==================

#%%
# Imports
# ---------

from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import MultiModelPerformanceReport

#%%
# Generate data & model
# -----------------------
iris = load_iris(as_frame=True)
train, test = train_test_split(iris.frame, test_size=0.33, random_state=42)

train_ds = Dataset(train, label="target")
test_ds = Dataset(test, label="target")

features = train_ds.data[train_ds.features]
label = train_ds.data[train_ds.label_name]
clf1 = AdaBoostClassifier().fit(features, label)
clf2 = RandomForestClassifier().fit(features, label)
clf3 = DecisionTreeClassifier().fit(features, label)

#%%
# Run the check
# ----------------
MultiModelPerformanceReport().run(train_ds, test_ds, [clf1, clf2, clf3])

#%%
# Regression check
# =================


#%%
# Imports
# --------
from sklearn.datasets import load_diabetes
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#%%
# Generate data & model
# -----------------------
diabetes = load_diabetes(as_frame=True)
train, test = train_test_split(diabetes.frame, test_size=0.33, random_state=42)

train_ds = Dataset(train, label="target", cat_features=['sex'])
test_ds = Dataset(test, label="target", cat_features=['sex'])

features = train_ds.data[train_ds.features]
label = train_ds.data[train_ds.label_name]
clf1 = AdaBoostRegressor().fit(features, label)
clf2 = RandomForestRegressor().fit(features, label)
clf3 = DecisionTreeRegressor().fit(features, label)

#%%
# Run the check
# ----------------
MultiModelPerformanceReport().run(train_ds, test_ds, [clf1, clf2, clf3])
