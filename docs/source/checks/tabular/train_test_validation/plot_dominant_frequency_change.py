# -*- coding: utf-8 -*-
"""
Dominant Frequency Change
*************************
This example provides an overview for using and understanding the `Dominant Frequency Change` check.

**Structure:**

* `What is a Dominant Frequency Change? <#what-is-a-dominant-frequency-change>`__
* `Generate Data <#generate-data>`__
* `Run The Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

What is a Dominant Frequency Change?
====================================
Dominant Frequency Change is a data integrity check which simply checks whether dominant values have increased
significantly between test data and train data. Sharp changes in dominant values can indicate a problem with the
data collection or data processing pipeline (for example, a sharp incrase in a common null or constant value),
and will cause the model to fail to generalize well. This check goal is to catch these issues early in the pipeline.

This check compares the dominant values of each feature in the test data to the dominant values of the same feature in
the train data. If the ratio of the test to train dominant values is greater than a threshold, the check fails.
This threshold can be configured by specifying the `ratio_change_thres` parameter of the check.

The Definition of a Dominant Value
----------------------------------
The dominant value is defined as a value that is frequent in data at least more than ``dominance_ratio`` times from the
next most frequent value. The ``dominance_ratio`` is a configurable parameter of the check.

"""

from deepchecks.tabular.checks.train_test_validation import DominantFrequencyChange
from deepchecks.tabular.datasets.classification import iris

#%%
# Generate data
# =============
train_ds, test_ds = iris.load_data(data_format='Dataset', as_train_test=True)

#%%
# Introducing Duplicates in the Test Data
# -----------------------------------------

# make duplicates in the test data
test_ds.data.loc[test_ds.data.index % 2 == 0, 'petal length (cm)'] = 5.1
test_ds.data.loc[test_ds.data.index / 3 > 8, 'sepal width (cm)'] = 2.7

#%%
# Run The Check
# =============

check = DominantFrequencyChange()
check.run(test_ds, train_ds)

#%%
# Define a Condition
# ===================
check = DominantFrequencyChange()
check.add_condition_ratio_of_change_not_greater_than(0.1)
res = check.run(test_ds, train_ds)
res.show(show_additional_outputs=False)
