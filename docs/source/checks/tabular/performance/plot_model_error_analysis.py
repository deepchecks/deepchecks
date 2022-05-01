# -*- coding: utf-8 -*-
"""
Model Error Analysis
********************

This notebooks provides an overview for using and understanding the model error analysis check.

**Structure:**

* `What is Model Error Analysis? <#what-is-model-error-analysis>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is Model Error Analysis?
=============================
The check trains a regression model to predict the error of the user's model. Then, the features scoring the highest
feature importance for the error regression model are selected and the distribution of the error vs the feature
values is plotted. The check results are shown only if the error regression model manages to predict the error
well enough.

Run the check
=============
We will run the check on the adult dataset which can be downloaded from the
`UCI machine learning repository <http://archive.ics.uci.edu/ml>`_ and is also available in
`deepchecks.tabular.datasets`.
"""

from deepchecks.tabular.datasets.classification import adult
from deepchecks.tabular.checks import ModelErrorAnalysis

train_ds, test_ds = adult.load_data()
model = adult.load_fitted_model()

check = ModelErrorAnalysis(min_error_model_score=0.3)
result = check.run(train_ds, test_ds, model)
result

#%%
# Define a condition
# ==================
# We can define a condition that enforces that the relative difference between the weak and strong segments is not
# greater than a certain ratio, for example ratio of 0.05

check = check.add_condition_segments_performance_relative_difference_not_greater_than(0.05)
result = check.run(train_ds, test_ds, model)
result.show(show_additional_outputs=False)
