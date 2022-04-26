# -*- coding: utf-8 -*-
"""
Train Test Samples Mix
**********************
This notebook provides an overview for using and understanding the Train Test Samples Mix check.

**Structure:**

* `Why samples mix are unwanted? <#why-samples-mix-are-unwanted>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

Why samples mix are unwanted?
=============================
Samples mix is a situation when train and test have joint samples. While using the test dataset for evaluating the
model, if we have samples from the train dataset which was used for training the model, then the resulting metrics will
be biased and won't reflect the performance we will get in a real life scenario. Therefore, it's important to always
avoid samples mix.

Run the check
=============
We will run the check on the iris dataset.
"""

from deepchecks.tabular.datasets.classification import iris
from deepchecks.tabular.checks.methodology import TrainTestSamplesMix

# Create data with leakage from train to test
train_df, test_df = iris.load_data(data_format='Dataframe')
bad_test = test_df.append(train_df.data.iloc[[0, 1, 1, 2, 3, 4, 2, 2, 10]], ignore_index=True)

check = TrainTestSamplesMix()
result = check.run(test_dataset=bad_test, train_dataset=train_df)
result

# %%
# Define a condition
# ==================
# We can define a condition that enforces that the ratio of samples in test which appears in train is below a given
# amount, the default is `0.1`.
check = TrainTestSamplesMix().add_condition_duplicates_ratio_not_greater_than()
result = check.run(test_dataset=bad_test, train_dataset=train_df)
result.show(show_additional_outputs=False)
