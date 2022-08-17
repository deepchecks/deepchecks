# -*- coding: utf-8 -*-
"""
.. _plot_tabular_feature_feature_correlation:

Feature Feature Correlation
***************************

This notebook provides an overview for using and understanding the feature-feature correlation check.

This check computes the pairwise correlations between the features, potentially spotting pairs of features that are
highly correlated.

**Structure:**

* `How are The Correlations Calculated? <#how-are-the-correlations-calculated>`__
* `Load Data <#load-data>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

How are The Correlations Calculated?
====================================

This check works with 2 types of features: categorical and numerical, and uses a different method to calculate the
correlation for each combination of feature types:

1. numerical-numerical: `Pearson's correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__
2. numerical-categorical: `Correlation ratio <https://en.wikipedia.org/wiki/Correlation_ratio>`__
3. categorical-categorical: `Symmetric Theil's U <https://en.wikipedia.org/wiki/Uncertainty_coefficient>`__

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
# We load the Adult dataset, a dataset based on the 1994 US Census containing both numerical and categorical features.

ds = adult.load_data(as_train_test=False)

#%%
# Run the Check
# ===============================================

check = FeatureFeatureCorrelation()
check.run(ds)

# To display the results in an IDE like PyCharm, you can use the following code:
# check.run(ds).show()
# The result will be displayed in a new window.

#%%
# Define a Condition
# ==================
# Now we will define a condition on the maximum number of pairs that are correlated above a certain threshold.
# In this example, we will define a condition that the maximum number of pairs that are correlated above 0.8 is less
# than 3.

check = FeatureFeatureCorrelation()
check.add_condition_max_number_of_pairs_above_threshold(0.8, 3)
result = check.run(ds)
result.show(show_additional_outputs=False)

