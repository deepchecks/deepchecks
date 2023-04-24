# -*- coding: utf-8 -*-
"""
.. _nlp__single_dataset_performance:

Single Dataset Performance
*****************************
This notebook provides an overview for using and understanding single dataset performance check for NLP tasks.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
==================================
This check is designed for evaluating a model's performance on a labeled dataset based on a scorer or multiple scorers.

Scorers are a convention of sklearn to evaluate a model,
it is a function which accepts (model, X, y_true) and returns a float result which is the score.
A sklearn convention is that higher scores are better than lower scores. For additional details `see scorers
documentation <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_.

The default scorers that are used are F1, Precision, and Recall.
"""

#%%
# Generate data & model
# ======================

from deepchecks.nlp.datasets.classification.tweet_emotion import load_data, load_precalculated_predictions

_, test_dataset = load_data(data_format='TextData')
_, test_probas = load_precalculated_predictions(pred_format='probabilities')

#%%
# Run the check
# ==============
#
# You can select which scorers to use by passing either a list or a dict of scorers to the check,
# see :doc:`Metrics Guide </user-guide/general/metrics_guide>` for additional details.

from deepchecks.nlp.checks import SingleDatasetPerformance

check = SingleDatasetPerformance(scorers=['recall_per_class', 'precision_per_class', 'f1_macro', 'f1_micro'])
result = check.run(dataset=test_dataset, probabilities=test_probas)
result.show()

#%%
# Define a condition
# ===================
# We can define on our check a condition to validate that the different metric scores are above a certain threshold.
# Using the ``class_mode`` argument we can define select a sub set of the classes to use for the condition.
#
# Let's add a condition to the check and see what happens when it fails:

check.add_condition_greater_than(threshold=0.85, class_mode='all')
result = check.run(dataset=test_dataset, probabilities=test_probas)
result.show(show_additional_outputs=False)

#%%
# We detected that the Recall score is below specified threshold in at least one of the classes.

#%%
