# -*- coding: utf-8 -*-
"""
.. _plot_tabular_class_imbalance:

Class Imbalance
***************

This notebook provides an overview for using and understanding the Class Imbalance check.

**Structure:**

* `What is the Class Imbalance check <#what-is-the-class-imbalance-check>`__
* `Generate data <#generate-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__


What is the Class Imbalance check
====================================
The ``ClassImbalance`` check produces a distribution of the target variable.
An indication for an imbalanced dataset is an uneven distribution in label classes.

An imbalanced dataset poses its own challenges, namely learning the characteristics of
the minority label, scarce minority instances to train on (or test for) and defining the
right evaluation metric.

Albeit, there are many techniques to address these challenges, including artificially increasing
the minority sample size (by over-sampling or using SMOTE), drop instances from the majority class (under-sampling),
using regularization, and adjusting the label classes weights.
"""


# %%
# Imports
# =========
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ClassImbalance
from deepchecks.tabular.datasets.classification import lending_club

# %%
# Generate data
# ===============

df = lending_club.load_data(data_format='Dataframe', as_train_test=False)
dataset = Dataset(df, label='loan_status', features=['id', 'loan_amnt'], cat_features=[])

# %%
# Run the check
# =================
ClassImbalance().run(dataset)


# %%
# Skew the target variable and run the check
# --------------------------------------------

df.loc[df.sample(frac=0.7, random_state=0).index, 'loan_status'] = 1
dataset = Dataset(df, label='loan_status', features=['id', 'loan_amnt'], cat_features=[])
ClassImbalance().run(dataset)


# %%
# Define a condition
# ====================
# A manually defined ratio between the labels can also be set:
ClassImbalance().add_condition_class_ratio_less_than(0.15).run(dataset)

