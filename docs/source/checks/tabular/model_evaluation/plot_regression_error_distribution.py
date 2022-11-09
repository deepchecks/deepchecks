# -*- coding: utf-8 -*-
"""
.. _plot_tabular_regression_error_distribution:

Regression Error Distribution
*****************************
This notebook provides an overview for using and understanding the Regression Error Distribution check.

**Structure:**

* `What is the Regression Error Distribution check? <#what-is-the-regression-error-distribution-check>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__


What is the Regression Error Distribution check?
==================================================
The ``RegressionErrorDistribution`` check shows the distribution of the regression error,
and enables to set conditions on two of the distribution parameters: Systematic error and Kurtosis values.
Kurtosis is a measure of the shape of the distribution, helping us understand if the distribution is significantly
"wider" from the normal distribution, which may imply a certain cause of error deforming the normal shape.
Systematic error, otherwise known as the error bias, is the mean prediction error of the model.
"""

#%%
# Run the check
# =============

#%%
# Generate data & model
# ----------------------
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

diabetes_df = load_diabetes(return_X_y=False, as_frame=True).frame
train_df, test_df = train_test_split(diabetes_df, test_size=0.33, random_state=42)

clf = GradientBoostingRegressor(random_state=0)
clf.fit(train_df.drop('target', axis=1), train_df['target'])

#%%
# Run the check (normal distribution)
# ---------------------------------------
# Since the following distribution resembles the normal distribution, both
# the kurtosis value and the systematic error will be ~0.

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import RegressionErrorDistribution

test = Dataset(test_df, label='target', cat_features=['sex'])
check = RegressionErrorDistribution()
check.run(test, clf)

#%%
# Skewing the data & rerun the check
# ----------------------------------

test.data[test.label_name] = 150
check.run(test, clf)

#%%
# Define a condition
# ==================
# After artificially skewing the target variable, both the kurtosis value and the systematic error
# would be significantly larger. In the conditions below we check if the systemic error, otherwise the mean prediction
# error, is less than 0.01 times the model's rmse score and that the kurtosis is greater than -0.1.

check = RegressionErrorDistribution()
check.add_condition_kurtosis_greater_than(threshold=-0.1)
check.add_condition_systematic_error_ratio_to_rmse_less_than(max_ratio=0.01)
check.run(test, clf)
