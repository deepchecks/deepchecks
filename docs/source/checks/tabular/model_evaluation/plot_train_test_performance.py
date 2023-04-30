# -*- coding: utf-8 -*-
"""
.. _tabular__train_test_performance:

Train Test Performance
***********************
This notebook provides an overview for using and understanding the train test performance check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__
* `Using a custom scorer <#using-a-custom-scorer>`__

What is the purpose of the check?
==================================
This check helps you compare your model's performance between the train and test datasets based on multiple scorers.

Scorers are a convention of sklearn to evaluate a model,
it is a function which accepts (model, X, y_true) and returns a float result which is the score.
A sklearn convention is that higher scores are better than lower scores. For additional details `see scorers
documentation <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`__.

The default scorers that are used are F1, Precision, and Recall for Classification
and Negative Root Mean Square Error, Negative Mean Absolute Error, and R2 for Regression.
"""

#%%
# Generate data & model
# ======================

from deepchecks.tabular.datasets.classification.iris import load_data, load_fitted_model

train_dataset, test_dataset = load_data()
model = load_fitted_model()

#%%
# Run the check
# ==============
#
# You can select which scorers to use by passing either a list or a dict of scorers to the check,
# the full list of possible scorers can be seen at the :ref:`metrics_user_guide`.

from deepchecks.tabular.checks import TrainTestPerformance

check = TrainTestPerformance(scorers=['recall_per_class', 'precision_per_class', 'f1_macro', 'f1_micro'])
result = check.run(train_dataset, test_dataset, model)
result.show()

#%%
# Define a condition
# ===================
# We can define on our check a condition that will validate that our model doesn't degrade
# on new data.
#
# Let's add a condition to the check and see what happens when it fails:

check.add_condition_train_test_relative_degradation_less_than(0.15)
result = check.run(train_dataset, test_dataset, model)
result.show(show_additional_outputs=False)

#%%
# We detected that for class "2" the Recall score result is degraded by more than 15%

#%%
# Using a custom scorer
# =======================
# In addition to the built-in scorers, we can define our own scorer based on sklearn api
# and run it using the check alongside other scorers:

from sklearn.metrics import fbeta_score, make_scorer

fbeta_scorer = make_scorer(fbeta_score, labels=[0, 1, 2], average=None, beta=0.2)

check = TrainTestPerformance(scorers={'my scorer': fbeta_scorer, 'recall': 'recall_per_class'})
result = check.run(train_dataset, test_dataset, model)
result.show()
