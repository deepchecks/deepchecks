# -*- coding: utf-8 -*-
"""
.. _plot_tabular_performance_report:

Performance Report
******************
This notebooks provides an overview for using and understanding performance report check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__
* `Using alternative scorers <#using-alternative-scorers>`__

What is the purpose of the check?
=================================
This check helps you compare your model's performance between two datasets.
The default metric that are used are F1, Percision, and Recall for Classification
and Negative Root Mean Square Error, Negative Mean Absolute Error, and R2 for Regression. RMSE and MAE Scorers are
negative because we subscribe to the sklearn convention of defining scoring functions.
`See scorers documentation <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_

"""

#%%
# Generate data & model
# =====================

from deepchecks.tabular.datasets.classification.phishing import (
    load_data, load_fitted_model)

train_dataset, test_dataset = load_data()
model = load_fitted_model()

#%%
# Run the check
# =============

from deepchecks.tabular.checks import PerformanceReport

check = PerformanceReport()
check.run(train_dataset, test_dataset, model)

#%%
# Define a condition
# ==================
# We can define on our check a condition that will validate that our model doesn't degrade
# on new data.
#
# Let's add a condition to the check and see what happens when it fails:

check = PerformanceReport()
check.add_condition_train_test_relative_degradation_less_than(0.05)
result = check.run(train_dataset, test_dataset, model)
result.show(show_additional_outputs=False)

#%%
# We detected that for class "0" our the Precision result is degraded by more than 5%

#%%
# Using alternative scorers
# =========================
# We can define alternative scorers that are not run by default:

from sklearn.metrics import fbeta_score, make_scorer

fbeta_scorer = make_scorer(fbeta_score, labels=[0, 1], average=None, beta=0.2)

check = PerformanceReport(alternative_scorers={'my scorer': fbeta_scorer})
check.run(train_dataset, test_dataset, model)
