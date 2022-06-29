# -*- coding: utf-8 -*-
"""
Weak Segment Performance
****************

This notebooks provides an overview for using and understanding the weak segment performance check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================
The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
its performance on different segments of your data. Specifically, it is designed to help you identify the model weakest
segments in the data distribution for further improvement and visibility purposes.

In order to achieve this, the check trains several simple tree based model to try and predict the user provided
model error. The relevant segments are detected by analyzing the different leafs of the trained trees.
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
# The check have several key parameters that can define the behavior of the check and especially its output.
#
# alternative_scorer: will define the relevant metric to use for the check. It is important to select a metric
# that is relevant to the data domain and task you are performing.
#
# segment_minimum_size_ratio: determines the minimum size of segments that are of interest. The check is tuned to find
# the weakest segment regardless of the segment size and so it is recommended to try different configurations of this
# parameter as larger segments can be of interest even the model performance on them is superior.
#
# categorical_aggregation_threshold: by default the check will combined rare categories into a single category called
# "other". This parameter determines the frequency threshold to be mapped into to the "other" category.
#

from deepchecks.tabular.datasets.classification import phishing
from deepchecks.tabular.checks import WeakSegmentsPerformance
from sklearn.metrics import make_scorer, f1_score

scorer = {'f1': make_scorer(f1_score, average='micro')}
_, test = phishing.load_data()
model = phishing.load_fitted_model()
check = WeakSegmentsPerformance(alternative_scorer=scorer, segment_minimum_size_ratio=0.03)
result = check.run(test, model)
result

#%%
# Observe the check's output
# --------------------------
# We see in the results that the check indeed found several segments on which the model performance is below average.
# In the heatmap display we can see model performance on the weakest segments and their environment with respect to the
# two features that are relevant to the segment. In order to get the full list of weak segments found we will inspect
# the result.value attribute.


result.value['weak_segments_list']

#%%
# Define a condition
# ==================
# We can define on our check a condition that will validate that the model performance on the weakest segment detected
# is grater then a specified ratio of the average model performance of the entire dataset.

# Let's add a condition and re-run the check:

check = WeakSegmentsPerformance(alternative_scorer=scorer, segment_minimum_size_ratio=0.03)
check.add_condition_segments_performance_relative_difference_greater_than(0.1)
result = check.run(train_dataset, test_dataset, model)
result.show(show_additional_outputs=False)
