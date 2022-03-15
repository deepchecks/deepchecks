# -*- coding: utf-8 -*-
"""
Whole Dataset Drift
*******************
This notebooks provides an overview for using and understanding the whole
dataset drift check.

**Structure:**

* `What is a dataset drift? <#what-is-a-dataset-drift>`__
* `Loading the Data <#loading-the-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is a dataset drift?
========================
A whole dataset drift, or a multivariate dataset drift, occurs when the
statistical properties of our input feature change, denoted by a change
in the distribution P(X).

Causes of data drift include:

* Upstream process changes, such as a sensor being replaced that changes
  the units of measurement from inches to centimeters.
* Data quality issues, such as a broken sensor always reading 0.
* Natural drift in the data, such as mean temperature changing with the seasons.
* Change in relation between features, or covariate shift.

The difference between a feature drift
<https://docs.deepchecks.com/en/stable/examples/checks/distribution/train_test_feature_drift.html>
(or univariate dataset drift) and a multivariate drift is that in the
latter the data drift occures in more that one feature.

In the context of machine learning, drift between the training set and the
test means that the model was trained on data that is different from the
current test data, thus it will probably make more mistakes predicting the
target variable.

How deepchecks detects dataset drift
------------------------------------
There are many methods to detect feature drift. Some of them are statistical
methods that aim to measure difference between distribution of 2 given sets.
This methods are more suited to univariate distributions and are primarily
used to detect drift between 2 subsets of a single feature.

Measuring a multivariate data drift is a bit more challenging. In the whole
dataset drift check, the multivariate drift is measured by training a classifier
that detects which samples come from a known distribution and defines the
drift by the accuracy of this classifier.

Practically, the check concatanates the train and the test sets, and assigns
label 0 to samples that come from the training set, and 1 to those who are
from the test set. Then, we train a binary classifer of type
`Histogram-based Gradient Boosting Classification Tree
<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>`__, and measure the
drift score from the AUC score of this classifier.
"""

#%%
# Loading the Data
# ================
# The dataset is the adult dataset which can be downloaded from the UCI machine learning repository.
#
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
# Irvine, CA: University of California, School of Information and Computer Science.

import pandas as pd
from urllib.request import urlopen
from sklearn.preprocessing import LabelEncoder
import numpy as np

name_data = urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names')
lines = [l.decode("utf-8") for l in name_data if ':' in l.decode("utf-8") and '|' not in l.decode("utf-8")]

features = [l.split(':')[0] for l in lines]
label_name = 'income'

cat_features = [l.split(':')[0] for l in lines if 'continuous' not in l]

train_df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                       names=features + [label_name])
test_df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                      names=features + [label_name], skiprows=1)

test_df[label_name] = test_df [label_name].str[:-1]

encoder = LabelEncoder()
encoder.fit(train_df[label_name])
train_df[label_name] = encoder.transform(train_df[label_name])
test_df[label_name] = encoder.transform(test_df[label_name])

#%%
# Process into dataset
# --------------------

from deepchecks.tabular import Dataset

cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 
                'race', 'sex', 'native-country']
train_ds = Dataset(train_df, label=label_name, cat_features=cat_features)
test_ds = Dataset(test_df, label=label_name, cat_features=cat_features)

numeric_features = list(set(train_ds.features) - set(cat_features))

#%%

train_ds.label_name

#%%
# Run the check
# =============
from deepchecks.tabular.checks import WholeDatasetDrift

check = WholeDatasetDrift()
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

train_drifted_ds = Dataset(train_drifted_df, label=label_name, cat_features=cat_features)
test_drifted_ds = Dataset(test_drifted_df, label=label_name, cat_features=cat_features)

#%%

check = WholeDatasetDrift()
check.run(train_dataset=train_drifted_ds, test_dataset=test_drifted_ds)

#%%
# As expected, the check detects a multivariate drift between the train and the
# test sets. It also displays the sex feature's distribution - the feature that
# contributed the most to that drift. This is reasonable since the sampling
# was biased based on that feature.
#
# Define a condition
# ==================
# Now, we define a condition that enforce the whole dataset drift score must be
# below 0.1. A condition is deepchecks' way to validate model and data quality,
# and let you know if anything goes wrong.

check = WholeDatasetDrift()
check.add_condition_overall_drift_value_not_greater_than(0.1)
check.run(train_dataset=train_drifted_ds, test_dataset=test_drifted_ds)

#%%
# As we see, our condition successfully detects the drift score is above the defined threshold.
