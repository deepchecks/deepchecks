# -*- coding: utf-8 -*-
"""
.. _plot_tabular_data_duplicates:

Data Duplicates
***************

This notebook provides an overview for using and understanding the data duplicates check:

**Structure:**

* `Why data duplicates? <#why-data-duplicates>`__
* `Load Data <#load-data>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__
"""

#%%

from datetime import datetime

import pandas as pd

from deepchecks.tabular.datasets.classification.phishing import load_data

#%%
# Why data duplicates?
# ====================
# The ``DataDuplicates`` check finds multiple instances of identical samples in the
# Dataset. Duplicate samples increase the weight the model gives to those samples.
# If these duplicates are there intentionally (e.g. as a result of intentional
# oversampling, or due to the dataset's nature it has identical-looking samples)
# this may be valid, however if this is an hidden issue we're not expecting to occur,
# it may be an indicator for a problem in the data pipeline that requires attention.
#
# Load Data
# =========


phishing_dataset = load_data(as_train_test=False, data_format='DataFrame')
phishing_dataset

#%%
# Run the Check
# =============

from deepchecks.tabular.checks import DataDuplicates

DataDuplicates().run(phishing_dataset)

# With Check Parameters
# ---------------------
# ``DataDuplicates`` check can also use a specific subset of columns (or alternatively
# use all columns except specific ignore_columns to check duplication):

DataDuplicates(columns=["entropy", "numParams"]).run(phishing_dataset)

#%%

DataDuplicates(ignore_columns=["scrape_date"], n_to_show=10).run(phishing_dataset)

#%%
# Define a Condition
# ==================
# Now, we define a condition that enforce the ratio of duplicates to be 0. A condition
# is deepchecks' way to validate model and data quality, and let you know if anything
# goes wrong.

check = DataDuplicates()
check.add_condition_ratio_less_or_equal(0)
result = check.run(phishing_dataset)
result.show(show_additional_outputs=False)
