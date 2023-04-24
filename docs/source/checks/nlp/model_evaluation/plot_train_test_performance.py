# -*- coding: utf-8 -*-
"""
.. _plot_nlp_train_test_performance:

Train Test Performance for NLP Models
**************************************
This notebook provides an overview for using and understanding train test performance check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data & predictions <#generate-data-predictions>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__
* `Using a custom scorer <#using-a-custom-scorer>`__

What is the purpose of the check?
==================================
This check helps you compare your NLP model's performance between the train and test datasets based on multiple metrics.

For Text Classification tasks the supported metrics are sklearn scorers. You may use any of the existing sklearn
scorers or create your own. For more information about the supported sklearn scorers, defining your own metrics and
to learn how to use metrics for other supported task types, see the
:doc:`Metrics Guide </user-guide/general/metrics_guide>`.

The default scorers that are used for are F1, Precision, and Recall for Classification,
and F1 Macro, Recall Macro and Precision Macro for Token Classification. See more about the supported task types at the
:doc:`Supported Tasks </user-guide/nlp/supported_tasks>` guide.
"""
import numpy as np

#%%
# Load data & predictions
# =======================

from deepchecks.nlp.datasets.classification.tweet_emotion import load_data, load_precalculated_predictions

train_dataset, test_dataset = load_data()
train_preds, test_preds = load_precalculated_predictions('predictions')

#%%
# Run the check
# ==============
#
# You can select which scorers to use by passing either a list or a dict of scorers to the check,
# the full list of possible scorers can be seen at scorers.py.

from deepchecks.nlp.checks import TrainTestPerformance

check = TrainTestPerformance(scorers=['recall_per_class', 'precision_per_class', 'f1_macro', 'f1_micro'])
result = check.run(train_dataset, test_dataset, train_predictions=train_preds, test_predictions=test_preds)
result.show()

#%%
# Define a condition
# ===================
# We can define on our check a condition that will validate that our model doesn't degrade
# on new data.
#
# Let's add a condition to the check and see what happens when it fails:

check.add_condition_train_test_relative_degradation_less_than(0.15)
result = check.run(train_dataset, test_dataset, train_predictions=train_preds, test_predictions=test_preds)
result.show(show_additional_outputs=False)

#%%
# We detected that for class "optimism" the Recall has degraded by more than 70%!

#%%
# Using a custom scorer
# =======================
# In addition to the built-in scorers, we can define our own scorer based on sklearn api
# and run it using the check alongside other scorers:

from sklearn.metrics import fbeta_score, make_scorer

fbeta_scorer = make_scorer(fbeta_score, labels=np.arange(len(set(test_dataset.label))), average=None, beta=0.2)

check = TrainTestPerformance(scorers={'my scorer': fbeta_scorer, 'recall': 'recall_per_class'})
result = check.run(train_dataset, test_dataset, train_predictions=train_preds, test_predictions=test_preds)
result.show()
