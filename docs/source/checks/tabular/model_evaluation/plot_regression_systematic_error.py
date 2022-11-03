# -*- coding: utf-8 -*-
"""
.. _plot_tabular_regression_systematic_error:

Regression Systematic Error
***************************
This notebook provides an overview for using and understanding the Regression Systematic Error check.

**This check is deprecated and will be removed in future versions**, please use the :doc:`Regression Error Distribution
</checks_gallery/tabular/model_evaluation/plot_regression_error_distribution>` check instead.

**Structure:**

* `What is the the Regression Systematic Error check? <#what-is-the-regression-systematic-error-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__


What is the Regression Systematic Error check?
==================================================
The ``RegressionSystematicError`` check looks for a systematic error in model predictions.
If the errors distribution is non-zero mean, it indicates the presence of a systematic error.

"""

#%%
# Imports
# =======

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import RegressionSystematicError

#%%
# Generate data & model
# ======================

diabetes_df = load_diabetes(return_X_y=False, as_frame=True).frame
train_df, test_df = train_test_split(diabetes_df, test_size=0.33, random_state=42)
train_df['target'] = train_df['target'] + 150

train = Dataset(train_df, label='target', cat_features=['sex'])
test = Dataset(test_df, label='target', cat_features=['sex'])

clf = GradientBoostingRegressor(random_state=0)
_ = clf.fit(train.data[train.features], train.data[train.label_name])

#%%
# Run the check
# ==============

check = RegressionSystematicError()
check.run(test, clf)
