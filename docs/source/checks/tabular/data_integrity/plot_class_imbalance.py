# -*- coding: utf-8 -*-
"""
.. _plot_tabular_class_imbalance:

Class Imbalance
***************

This notebook provides an overview for using and understanding the Class Imbalance check.

**Structure:**

* `What is the Class Imbalance check <#what-is-the-class-imbalance-check>`__
* `Generate data <#generate-datal>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__
"""

# %%
# What is the Class Imbalance check
# ====================================
# The ``ClassImbalance`` check produces a distribution of the target variable.
# An indication for an imbalanced dataset is a biased class distribution across
# one or multiple labels.

# An imbalanced dataset posses its own challenges, namely learning the characteristics of
# the minority label, scarce minority instances for both train and test and defining the
# right evaluation metric.

# Albeit, there are many techniques to address with these challenges, including artificially increasing
# the minority sample size with SMOTE (over-sampling), drop instances from the majority class (under-sampling),
# use regularization and adjust the label weight.


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

