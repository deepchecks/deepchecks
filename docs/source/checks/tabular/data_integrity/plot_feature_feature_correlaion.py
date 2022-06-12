# -*- coding: utf-8 -*-
"""
Feature Feature Correlation
***************************
This notebooks provides an overview for using and understanding the feature-feature correlation check.

This check computes the pairwise correlations between the features, potentially spotting pairs of features that are
highly correlated.

**Structure:**

* `How are The Correlations Calculated? <#how-are-the-correlations-calculated>`__
* `Load Data <#load-data>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

How are The Correlations Calculated?
============================
This check works with 2 types of features: categorical and numerical, and used a different method to calculate the
correlation for each combination of feature types.

1. numerical-numerical: Pearson's correlation coefficient
2. numerical-categorical: Correlation ratio
3. categorical-categorical: Symmetric Theil's U statistic

"""

#%%
# Imports
# =======

import pandas as pd
from deepchecks.tabular.datasets.classification import adult
from deepchecks.tabular.checks.data_integrity import FeatureFeatureCorrelation

#%%
# Load Data
# ===============

ds = adult.load_data(as_train_test=False)

#%%
# Run the Check
# ===============================================

check = FeatureFeatureCorrelation()
check.run(ds)

# For displaying the results in an IDE like PyCharm, you can use the following code:
# check.run(ds).show()
# The result will be displayed in a new window.

#%%
# Define a Condition
# ==================
# Now we will define a condition on the maximum number of pairs that are correlated above a certain threshold.
# In this example, we will define a condition that the maximum number of pairs that are correlated above 0.8 is less
# than 3.

check = FeatureFeatureCorrelation()
check.add_condition_max_number_of_pairs_above(0.8, 3)
result = check.run(ds)
result.show(show_additional_outputs=False)

