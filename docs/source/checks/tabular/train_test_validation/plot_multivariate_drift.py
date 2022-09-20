# -*- coding: utf-8 -*-
"""
.. _plot_tabular_multivariate_drift:

Multivariate Drift
*******************

This notebooks provides an overview for using and understanding the multivariate
drift check.

**Structure:**

* `What Is Multivariate Drift? <#what-is-a-multivariate-drift>`__
* `Loading the Data <#loading-the-data>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

What Is Multivariate Drift?
==============================

Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

A multivariate drift is a drift that occurs in more than one feature at a time,
and may even affect the relationships between those features, which are undetectable by
univariate drift methods.
The multivariate drift check tries to detect multivariate drift between the two input datasets.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide>`.

How Deepchecks Detects Dataset Drift
------------------------------------

This check detects multivariate drift by using :ref:`a domain classifier <drift_detection_by_domain_classifier>`.
Other methods to detect drift include :ref:`univariate measures <drift_detection_by_univariate_measure>`
which is used in other checks, such as :doc:`Train Test Feature Drift check </checks_gallery/tabular/train_test_validation/plot_train_test_feature_drift>`.
"""

#%%
# Loading the Data
# ================
# The dataset is the adult dataset which can be downloaded from the UCI machine learning repository.
#
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
# Irvine, CA: University of California, School of Information and Computer Science.

from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from deepchecks.tabular import Dataset
from deepchecks.tabular.datasets.classification import adult

#%%
# Create Dataset
# ==============

label_name = 'income'
train_ds, test_ds = adult.load_data()
encoder = LabelEncoder()
train_ds.data[label_name] = encoder.fit_transform(train_ds.data[label_name])
test_ds.data[label_name] = encoder.transform(test_ds.data[label_name])

#%%

train_ds.label_name

#%%
# Run the Check
# =============
from deepchecks.tabular.checks import MultivariateDrift

check = MultivariateDrift()
check.run(train_dataset=train_ds, test_dataset=test_ds)

#%%
# We can see that there is almost no drift found between the train and the test
# set of the raw adult dataset. In addition to the drift score the check displays
# the top features that contibuted to the data drift.
#
# Introduce drift to dataset
# ==========================
# Now, let's try to add a manual data drift to the data by sampling a biased
# portion of the training data

sample_size = 10000
random_seed = 0

#%%

train_drifted_df = pd.concat([train_ds.data.sample(min(sample_size, train_ds.n_samples) - 5000, random_state=random_seed), 
                             train_ds.data[train_ds.data['sex'] == ' Female'].sample(5000, random_state=random_seed)])
test_drifted_df = test_ds.data.sample(min(sample_size, test_ds.n_samples), random_state=random_seed)

train_drifted_ds = Dataset(train_drifted_df, label=label_name, cat_features=train_ds.cat_features)
test_drifted_ds = Dataset(test_drifted_df, label=label_name, cat_features=test_ds.cat_features)

#%%

check = MultivariateDrift()
check.run(train_dataset=train_drifted_ds, test_dataset=test_drifted_ds)

#%%
# As expected, the check detects a multivariate drift between the train and the
# test sets. It also displays the sex feature's distribution - the feature that
# contributed the most to that drift. This is reasonable since the sampling
# was biased based on that feature.
#
# Define a Condition
# ==================
# Now, we define a condition that enforce the multivariate drift score must be
# below 0.1. A condition is deepchecks' way to validate model and data quality,
# and let you know if anything goes wrong.

check = MultivariateDrift()
check.add_condition_overall_drift_value_less_than(0.1)
check.run(train_dataset=train_drifted_ds, test_dataset=test_drifted_ds)

#%%
# As we see, our condition successfully detects the drift score is above the defined threshold.
