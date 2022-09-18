# -*- coding: utf-8 -*-
"""
.. _plot_tabular_regression_error_distribution:

Regression Error Distribution
*****************************
This notebook provides an overview for using and understanding the Regression Error Distribution check.

**Structure:**

* `What is the Regression Error Distribution check? <#what-is-the-regression-error-distribution-check>`__
* `Run the check - normal distribution <#run-the-check-normal-distribution>`__
* `Run the check - abnormal distribution <#run-the-check-abnormal-distribution>`__


What is the Regression Error Distribution check?
==================================================
The ``RegressionErrorDistribution`` check shows the distribution of the regression error,
and enables to set conditions on the distribution's kurtosis. Kurtosis is a measure of the shape
of the distribution, helping us understand if the distribution is significantly "wider" from
the normal distribution, which may imply a certain cause of error deforming the normal shape.

"""

#%%
# Imports
# ========

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import RegressionErrorDistribution

#%%
# Run the check - normal distribution
# ====================================

#%%
# Generate data & model
# ----------------------

diabetes_df = load_diabetes(return_X_y=False, as_frame=True).frame
train_df, test_df = train_test_split(diabetes_df, test_size=0.33, random_state=42)

train = Dataset(train_df, label='target', cat_features=['sex'])
test = Dataset(test_df, label='target', cat_features=['sex'])

clf = GradientBoostingRegressor(random_state=0)
_ = clf.fit(train.data[train.features], train.data[train.label_name])

#%%
# Run the check (normal distribution)
# ---------------------------------------
# Since the following distribution resembles the normal distribution,
# the kurtosis will be ~0.
check = RegressionErrorDistribution()
check.run(test, clf)


#%%
# Skewing the data
# ------------------

test.data[test.label_name] = 150


#%%
# Run the check - abnormal distribution
# =======================================

check = RegressionErrorDistribution()
check.run(test, clf)

#%%
# Define a condition
# --------------------
# Since we artificially skewed the target variable, the kurtosis would be bigger than
# the previous check, which implies a certain cause of error.

check = RegressionErrorDistribution()
check.add_condition_kurtosis_greater_than(0.1).run(test, clf)
