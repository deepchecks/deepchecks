# -*- coding: utf-8 -*-
"""
Metadata Segments Performance
*************************

This notebook provides an overview for using and understanding the metadata segment performance check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Automatically detecting weak segments <#automatically-detecting-weak-segments>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
==================================

The check is designed to help you easily identify the model's weakest segments based on the provided
:func:`metadata <deepchecks.nlp.text_data.TextData.set_metadata>`. In addition,
it enables to provide a sublist of the metadata columns, thus limiting the check to search in
interesting subspaces.

Automatically detecting weak segments
=====================================

The check contains several steps:

#. We calculate loss for each sample in the dataset using the provided model via either log-loss or MSE according
   to the task type.

#. Select a subset of features for the weak segment search. This is done by selecting the features with the
   highest feature importance to the model provided (within the features selected for check, if limited).

#. We train multiple simple tree based models, each one is trained using exactly two
   features (out of the ones selected above) to predict the per sample error calculated before.

#. We convert each of the leafs in each of the trees into a segment and calculate the segment's performance. For the
   weakest segments detected we also calculate the model's performance on data segments surrounding them.
"""
#%%
# Generate data & model
# =====================

from deepchecks.nlp.datasets.classification.tweet_emotion import load_data, load_precalculated_predictions

_, test_dataset = load_data(data_format='TextData')
_, test_probas = load_precalculated_predictions(pred_format='probabilities')

test_dataset.metadata.head(3)

#%%
# Run the check
# =============
#
# The check has several key parameters (that are all optional) that affect the behavior of the
# check and especially its output.
#
# ``columns / ignore_columns``: Controls which columns should be searched for weak segments. By default,
# uses all columns.
#
# ``alternative_scorer``: Determines the metric to be used as the performance measurement of the model on different
# segments. It is important to select a metric that is relevant to the data domain and task you are performing.
#
# ``segment_minimum_size_ratio``: Determines the minimum size of segments that are of interest. The check is tuned
# to find the weakest segment regardless of the segment size and so it is recommended to try different configurations
# of this parameter as larger segments can be of interest even the model performance on them is superior.
#
# ``categorical_aggregation_threshold``: By default the check will combine rare categories into a single category called
# "Other". This parameter determines the frequency threshold for categories to be mapped into to the "other" category.
#
# see :class:`API reference <deepchecks.tabular.checks.model_evaluation.WeakSegmentsPerformance>` for more details.

from deepchecks.nlp.checks import MetadataSegmentsPerformance
from sklearn.metrics import make_scorer, f1_score

scorer = {'f1': make_scorer(f1_score, average='micro')}
check = MetadataSegmentsPerformance(alternative_scorer=scorer,
                                    segment_minimum_size_ratio=0.03)
result = check.run(test_dataset, probabilities=test_probas)
result.show()

#%%
# Observe the check's output
# --------------------------
#
# We see in the results that the check indeed found several segments on which the model performance is below average.
# In the heatmap display we can see model performance on the weakest segments and their environment with respect to the
# two features that are relevant to the segment. In order to get the full list of weak segments found we will inspect
# the result.value attribute.


result.value['weak_segments_list'].head(3)

#%%
# Define a condition
# ==================
#
# We can define on our check a condition that will validate that the model performance on the weakest segment detected
# is greater than a specified ratio of the average model performance of the entire dataset.

# Let's add a condition and re-run the check:

check = MetadataSegmentsPerformance(alternative_scorer=scorer, segment_minimum_size_ratio=0.03)
check.add_condition_segments_relative_performance_greater_than(0.1)
result = check.run(test_dataset, probabilities=test_probas)
result.show(show_additional_outputs=False)
