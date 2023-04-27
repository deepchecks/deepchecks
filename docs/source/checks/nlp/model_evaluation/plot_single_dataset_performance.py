# -*- coding: utf-8 -*-
"""
.. _nlp__single_dataset_performance:

Single Dataset Performance
*****************************
This notebook provides an overview for using and understanding the single dataset performance check for NLP tasks.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
==================================
This check is designed for evaluating a model's performance on a labeled dataset based on a scorer or multiple scorers.

For Text Classification tasks the supported metrics are sklearn scorers. You may use any of the existing sklearn
scorers or create your own. For more information about the supported sklearn scorers, defining your own metrics and
to learn how to use metrics for other supported task types, see the
:doc:`Metrics Guide </user-guide/general/metrics_guide>`.

The default scorers that are used for are F1, Precision, and Recall for Classification,
and F1 Macro, Recall Macro and Precision Macro for Token Classification. See more about the supported task types at the
:doc:`Supported Tasks </user-guide/nlp/supported_tasks>` guide.
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
# see :ref:`Metrics Guide <metrics_user_guide>` for additional details.

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
