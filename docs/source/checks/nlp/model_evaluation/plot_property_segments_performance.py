# -*- coding: utf-8 -*-
"""
.. _nlp__property_segments_performance:


Property Segments Performance
*****************************

This notebook provides an overview for using and understanding the property segment performance check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Automatically detecting weak segments <#automatically-detecting-weak-segments>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================

The check is designed to help you easily identify the model's weakest segments based on the provided
:ref:`properties <nlp__properties_guide>`. In addition,
it enables to provide a sublist of the metadata columns, thus limiting the check to search in
interesting subspaces.

Automatically detecting weak segments
=====================================

The check contains several steps:

#. We calculate loss for each sample in the dataset using the provided model via either log-loss or MSE according
   to the task type.

#. We train multiple simple tree based models, each one is trained using exactly two
   properties (out of the ones selected above) to predict the per sample error calculated before.

#. We extract the corresponding data samples for each of the leaves in each of the trees (data segments) and calculate
   the model performance on them. For the weakest data segments detected we also calculate the model's
   performance on data segments surrounding them.
"""
# %%
# Generate data & model
# =====================

from deepchecks.nlp.datasets.classification.tweet_emotion import load_data, load_precalculated_predictions

_, test_dataset = load_data(data_format="TextData")
_, test_probas = load_precalculated_predictions(pred_format="probabilities")

test_dataset.properties.head(3)

# %%
# Run the check
# =============
#
# The check has several key parameters (that are all optional) that affect the behavior of the
# check and especially its output.
#
# ``properties / ignore_properties``: Controls which properties should be searched for weak segments. By default,
# uses all properties data provided.
#
# ``alternative_scorer``: Determines the metric to be used as the performance measurement of the model on different
# segments. It is important to select a metric that is relevant to the data domain and task you are performing.
# For additional information on scorers and how to use them see
# :ref:`Metrics Guide <metrics_user_guide>`.
#
# ``segment_minimum_size_ratio``: Determines the minimum size of segments that are of interest. The check will
# return data segments that contain at least this fraction of the total data samples. It is recommended to
# try different configurations
# of this parameter as larger segments can be of interest even the model performance on them is superior.
#
# ``categorical_aggregation_threshold``: By default the check will combine rare categories into a single category called
# "Other". This parameter determines the frequency threshold for categories to be mapped into to the "other" category.
#
# ``multiple_segments_per_column``: If True, will allow the same property to be a segmenting feature in
# multiple segments,  otherwise each property can appear in one segment at most. False by default.
#
# see :class:`API reference <deepchecks.tabular.checks.model_evaluation.WeakSegmentsPerformance>` for more details.

from deepchecks.nlp.checks import PropertySegmentsPerformance

from sklearn.metrics import f1_score, make_scorer

scorer = {"f1": make_scorer(f1_score, average="micro")}
check = PropertySegmentsPerformance(alternative_scorer=scorer, segment_minimum_size_ratio=0.03)
result = check.run(test_dataset, probabilities=test_probas)
result.show()

# %%
# Observe the check's output
# --------------------------
#
# We see in the results that the check indeed found several segments on which the model performance is below average.
# In the heatmap display we can see model performance on the weakest segments and their environment with respect to the
# two features that are relevant to the segment. In order to get the full list of weak segments found we will inspect
# the ``result.value`` attribute. Shown below are the 3 segments with the worst performance.


result.value["weak_segments_list"].head(3)

# %%
# Define a condition
# ==================
#
# We can add a condition that will validate the model's performance on the weakest segment detected is above a certain
# threshold. A scenario where this can be useful is when we want to make sure that the model is not under performing
# on a subset of the data that is of interest to us.

# Let's add a condition and re-run the check:

check.add_condition_segments_relative_performance_greater_than(0.1)
result = check.run(test_dataset, probabilities=test_probas)
result.show(show_additional_outputs=False)
