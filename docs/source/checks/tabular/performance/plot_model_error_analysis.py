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
Evaluating the model's overall performance metrics gives a good high-level overview and can be useful for tracking model progress during training of for comparing models. However, when it's time to fully evaluate if a model is fit for production, or when you're interested in a deeper understanding of your model's performance in order to improve it or to be aware of its weaknesses, it's recommended
to look deeper at how the model performs on various segments of the data. The model error analysis check searches for data segments in which the model error is significantly lower from the model error of the dataset as a whole.

Algorithm:
----------

1. Computes the per-sample loss (for log-loss for classification, mse for regression).
2. Trains a regression model to predict the error of the user's model, based on the input features.
3. Repeat stage 2 several times with various tree parameters and random states to ensure that the most relevant partitions for model error are selected.
4. The features scoring the highest feature importance for the error regression model are selected and the distribution of the error vs the feature values is plotted.

The check results are shown only if the error regression model manages
to predict the error well enough. The resulting plots show the distribution of the error for the features that are
most effective at segmenting the error to high and low values, without need for manual selection of segmentation
features.

Related Checks:
---------------
When the important segments of the data are known in advance (when we know that some population segments have
different behaviours and business importance, for example income levels or state of residence) it is possible to just
have a look at the performance at various pre-defined segments. In deepchecks, this can be done using the
:doc:`Segment Performance </checks_gallery/tabular/performance/plot_segment_performance>`_ check, which shows the
performance for segments defined by combination of values from two pre-defined columns.


Run the check
=============
We will run the check on the adult dataset which can be downloaded from the
`UCI machine learning repository <http://archive.ics.uci.edu/ml>`_ and is also available in
`deepchecks.tabular.datasets`.
"""

from deepchecks.tabular.datasets.classification import adult
from deepchecks.tabular.checks import ModelErrorAnalysis

train_ds, test_ds = adult.load_data(data_format='Dataset', as_train_test=True)
model = adult.load_fitted_model()

check = ModelErrorAnalysis(min_error_model_score=0.3)
result = check.run(train_ds, test_ds, model)
result

#%%
# The check has found that the features 'hours-per-week', 'age' and 'relationship' are the most predictive of
# differences in the model error. We can further investigate the model performance by passing two of these columns
# to the :doc:`Segment Performance </checks_gallery/tabular/performance/plot_segment_performance>`_ check:

from deepchecks.tabular.checks import SegmentPerformance

SegmentPerformance(feature_1='age', feature_2='relationship').run(test_ds, model)

#%%
# From which we learn that the model error is exceptionally higher for people in the "Husband" or "Other" status,
# except for the lower age groups for which the error is lower.

#%%
# Define a condition
# ==================
# We can define a condition that enforces that the relative difference between the weak and strong segments is not
# greater than a certain ratio, for example ratio of 0.05

check = check.add_condition_segments_performance_relative_difference_not_greater_than(0.05)
result = check.run(train_ds, test_ds, model)
result.show(show_additional_outputs=False)
