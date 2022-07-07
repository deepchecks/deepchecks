# -*- coding: utf-8 -*-
"""
.. _plot_tabular_train_test_samples_mix:

Train Test Samples Mix
**********************
This notebook provides an overview for using and understanding the Train Test Samples Mix check.

**Structure:**

* `Why is samples mix unwanted? <#why-is-samples-mix-unwanted>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

Why is samples mix unwanted?
=============================
Samples mix is when the train and test datasets have some samples in common.
We use the test dataset in order to evaluate our model performance, and having samples in common with the train dataset
will lead to biased metrics, which does not represent the real performance we will get in a real scenario. Therefore,
we always want to avoid samples mix.

Run the check
=============
We will run the check on the iris dataset.
"""

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestSamplesMix
from deepchecks.tabular.datasets.classification import iris

# Create data with leakage from train to test
train, test = iris.load_data()
bad_test_df = test.data.append(train.data.iloc[[0, 1, 1, 2, 3, 4, 2, 2, 10]], ignore_index=True)
bad_test = test.copy(bad_test_df)

check = TrainTestSamplesMix()
result = check.run(test_dataset=bad_test, train_dataset=train)
result

# %%
# Define a condition
# ==================
# We can define a condition that enforces that the ratio of samples in test which appears in train is below a given
# amount, the default is `0.1`.
check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal()
result = check.run(test_dataset=bad_test, train_dataset=train)
result.show(show_additional_outputs=False)
