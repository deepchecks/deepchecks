# -*- coding: utf-8 -*-
"""
.. _quick_full_suite:

Quickstart - Full Suite in 5 Minutes
************************************

In order to run your first Deepchecks Suite all you need to have is the data
and model that you wish to validate. More specifically, you need:

* Your train and test data (in Pandas DataFrames or Numpy Arrays)
* (optional) A :doc:`supported model </user-guide/tabular/supported_models>` (including XGBoost,
  scikit-learn models, and many more). Required for running checks that need the
  model's predictions for running.

To run your first suite on your data and model, you need only a few lines of
code, that start here: `Define a Dataset Object <#define-a-dataset-object>`__.

# If you don't have deepchecks installed yet:

.. code:: python

    # If you don't have deepchecks installed yet:
    import sys
    !{sys.executable} -m pip install deepchecks -U --quiet #--user

"""

#%%
# Load Data, Split Train-Val, and Train a Simple Model
# ====================================================
# For the purpose of this guide we'll use the simple iris dataset and train a
# simple random forest model for multiclass classification:

import numpy as np
# General imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deepchecks.tabular.datasets.classification import iris

# Load Data
iris_df = iris.load_data(data_format='Dataframe', as_train_test=False)
label_col = 'target'
df_train, df_test = train_test_split(iris_df, stratify=iris_df[label_col], random_state=0)

# Train Model
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(df_train.drop(label_col, axis=1), df_train[label_col]);

#%%
# Define a Dataset Object
# =======================
# Initialize the Dataset object, stating the relevant metadata about the dataset
# (e.g. the name for the label column)
#
# Check out the Dataset's attributes to see which additional special columns can be
# declared and used (e.g. date column, index column).

from deepchecks.tabular import Dataset

# We explicitly state that this dataset has no categorical features, otherwise they will be automatically inferred
# If the dataset has categorical features, the best practice is to pass a list with their names

ds_train = Dataset(df_train, label=label_col, cat_features=[])
ds_test =  Dataset(df_test,  label=label_col, cat_features=[])

#%%
# Run a Deepchecks Suite
# ======================
# Run the full suite
# ------------------
# Use the ``full_suite`` that is a collection of (most of) the prebuilt checks.
#
# Check out the :doc:`when should you use </getting-started/when_should_you_use>`
# deepchecks guide for some more info about the existing suites and when to use them.

from deepchecks.tabular.suites import full_suite

suite = full_suite()

#%%

suite.run(train_dataset=ds_train, test_dataset=ds_test, model=rf_clf)

#%%
# Run the integrity suite
# -----------------------
# If you still haven't started modeling and just have a single dataset, you
# can use the ``data_integrity``:

from deepchecks.tabular.suites import data_integrity

integ_suite = data_integrity()
integ_suite.run(ds_train)

#%%
# Run a Deepchecks Check
# ======================
# If you want to run a specific check, you can just import it and run it directly.
#
# Check out the :doc:`Check tabular examples </checks_gallery/tabular/index>` in
# the examples or the :doc:`API Reference </api/index>` for more info about the
# existing checks and their parameters.

from deepchecks.tabular.checks import TrainTestLabelDrift

#%%

check = TrainTestLabelDrift()
result = check.run(ds_train, ds_test)
result

#%%
# and also inspect the result value which has a check-dependant structure:

result.value

#%%
# Edit an Existing Suite
# ======================
# Inspect suite and remove condition
# ----------------------------------
# We can see that the Feature Label Correlation check failed, both for test and for
# train. Since this is a very simple dataset with few features and this behavior
# is not necessarily problematic, we will remove the existing conditions for the PPS

# Lets first print the suite to find the conditions that we want to change:

suite

#%%

# now we can use the check's index and the condition's number to remove it:
print(suite[5])
suite[5].remove_condition(0)

#%%

# print and see that the condition was removed
suite[5]

#%%
# If we now re-run the suite, all of the existing conditions will pass.
#
# *Note: the check we manipulated will still run as part of the Suite, however
# it won't appear in the Conditions Summary since it no longer has any
# conditions defined on it. You can still see its display results in the 
# Additional Outputs section*
#
# **For more info about working with conditions, see the detailed configuring 
# conditions guide.**