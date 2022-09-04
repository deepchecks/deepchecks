# -*- coding: utf-8 -*-
"""
.. _plot_tabular_feature_label_correlation_change:

Feature Label Correlation Change
********************************

This notebook provides an overview for using and understanding the "Feature Label Correlation Change" check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data <#generate-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================
The check estimates for every feature its ability to predict the label by itself.
This check can help find:

* A potential leakage (between the label and a feature) in both datasets
  - e.g. due to incorrect sampling during data collection. This is a critical
  problem, that will likely stay hidden without this check (as it won't pop
  up when comparing model performance on train and test).
* A strong drift between the the feature-label relation in both datasets,
  possibly originating from a leakage in one of the datasets - e.g. a
  leakage that exists in the training data, but not necessarily in a
  "fresh" dataset, that may have been built differently.

The check is based on calculating the predictive power score (PPS) of each
feature. For more details you can read here `how the PPS is calculated 
<#how-is-the-predictive-power-score-pps-calculated>`__.

What is a problematic result?
-----------------------------
1. Features with a high predictive score - can indicate that there is a leakage
   between the label and the feature, meaning that the feature holds information
   that is somewhat based on the label to begin with.

   For example: a bank uses their loans database to create a model of whether
   a customer will be able to return a loan. One of the features they extract
   is "number of late payments". It is clear this feature will have a very
   strong prediction power on the customer's ability to return his loan,
   but this feature is based on data the bank knows only after the loan is
   given, so it won't be available during the time of the prediction, and is
   a type of leakage.
2. A high difference between the PPS scores of a certain feature in the train
   and in the test datasets - this is an indication for a drift between the
   relation of the feature and the label and a possible leakage in one of 
   the datasets.

   For example: a coffee shop chain trained a model to predict the number of
   coffee cups ordered in a store, and the model was trained on data from a
   specific state, and tested on data from all states. Running the Feature
   Label Correlation check on this split found that there was a high
   difference in the PPS score of the feature "time_in_day" - it had a
   much higher predictive power on the training data than on the test data.
   Investigating this topic led to detection of the problem - the time in
   day was saved in UTC time for all states, which made the feature much
   less indicative for the test data as it had data from several time
   zones (and much more coffee cups are ordered in during the morning/noon
   than during the evening/night time). This was fixed by changing the
   feature to be the time relative to the local time zone, thus fixing its
   predictive power and improving the model's overall performance.

.. _plot_tabular_feature_label_correlation_change__how_is_the_predictive_power_score_pps_calculated:

How is the Predictive Power Score (PPS) calculated?
===================================================
The features' predictive score results in a numeric score between 0 (feature
has no predictive power) and 1 (feature can fully predict the label alone).

The process of calculating the PPS is the following:
"""

#%%
# 1. Extract from the data only the label and the feature being tested
# 2. Drop samples with missing values
# 3. Keep 5000 (this is configurable parameter) samples from the data
# 4. Preprocess categorical columns. For the label using ``sklearn.LabelEncoder`` and for the feature using ``sklearn.OneHotEncoder``
# 5. Partition the data with 4-fold cross-validation
# 6. Train decision tree
# 7. Compare the trained model's performance with naive model's performance as follows:
#
# Regression: The naive model always predicts the median of the label column,
# the metric being used is MAE and the PPS calculation is: :math:`1 - \frac{\text{MAE model}}{\text{MAE naive}}`
#
# Classification: The naive model always predicts the most common class of
# the label column, The metric being used is F1 and the PPS calculation is:
# :math:`\frac{\text{F1 model} - \text{F1 naive}}{1 - \text{F1 naive}}`
#
# .. note::
#
#    All the PPS parameters can be changed by passing to the check the parameter ``ppscore_params``
#
# For further information about PPS you can visit the `ppscore github
# <https://github.com/8080labs/ppscore>`__ or the following blog post: `RIP correlation.
# Introducing the Predictive Power Score
# <https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598>`__


#%%
# Generate data
# =============
# We'll add to a given dataset a direct relation between two features and the label,
# in order to see the Feature Label Correlation Change check in action.

from deepchecks.tabular.datasets.classification.phishing import load_data


def relate_column_to_label(dataset, column, label_power):
    col_data = dataset.data[column]
    dataset.data[column] = col_data + (dataset.data[dataset.label_name] * col_data.mean() * label_power)
    
train_dataset, test_dataset = load_data()

# Transforming 2 features in the dataset given to add correlation to the label 
relate_column_to_label(train_dataset, 'numDigits', 10)
relate_column_to_label(train_dataset, 'numLinks', 10)
relate_column_to_label(test_dataset, 'numDigits', 0.1)

#%%
# Run the check
# =============
from deepchecks.tabular.checks import FeatureLabelCorrelationChange

result = FeatureLabelCorrelationChange().run(train_dataset=train_dataset, test_dataset=test_dataset)
result

#%%
# Observe the check's output
# --------------------------
# The check shows the top features with the highest PPS difference in the datasets,
# and elaborates how to interpret the results. By default only the top 5 features
# are displayed, it can be changed by using the parameter ``n_show_top`` of the check.
#
# In addition to the graphic output, the check also returns a value which includes
# all of the information that is needed for defining the conditions for validation.
# The value is a dictionary of:
#
# * train - for train dataset for each column the numeric PPS score (0 to 1)
# * test - for test dataset for each column the numeric PPS score (0 to 1)
# * train-test difference - for each column the difference between the datasets scores, as ``train - test``

result.value

#%%
# Define a condition
# ==================
# We can define on our check a condition that will validate that our pps scores aren't
# too high. The check has 2 possible built-in conditions:
# ``add_condition_feature_pps_difference_not_greater_than`` - Validate that the difference
# in the PPS between train and test is not larger than defined amount (default 0.2)
#
# ``add_condition_feature_pps_in_train_not_greater_than`` - Validate that the PPS scores on
# train dataset are not exceeding a defined amount (default 0.7)
#
# Let's add the conditions, and re-run the check:

check = FeatureLabelCorrelationChange().add_condition_feature_pps_difference_less_than().add_condition_feature_pps_in_train_less_than()
result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
result.show(show_additional_outputs=False)
