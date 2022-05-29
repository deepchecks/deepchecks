# -*- coding: utf-8 -*-
"""
Train Test Validation Scenario on Lending Club Data - 5 min Quickstart
***********************************************************************

The deepchecks train test validation suite is relevant any time you have data splits
that you wish to validate:
whether it's for comparing distributions across different data splits
(e.g. before training a model or when splitting data for cross-validation),
or for comparing new data batch to previous data batches.
Here we'll use data from the lending club dataset, to demonstrate how you can run
the suite with only a few simple lines of code, and see which kind of insights it can find.

.. code-block:: bash

    # Before we start, if you don't have deepchecks installed yet,
    # make sure to run:
    pip install deepchecks -U --quiet #--user
"""

#%%
# Load Data and Prepare Data
# ====================================================
#
# Load Data
# -----------


from deepchecks.tabular import datasets
import pandas as pd

# TODO - load this from datasets...
# data = datasets.regression.avocado.load_data(data_format='DataFrame', as_train_test=False)
data = pd.read_csv("lc-2017-18.csv", parse_dates=['issue_d'])
data.head(2)


#%%
# Define Lending Club Metadata
# ------------------------------

categorical_features = ['addr_state', 'application_type', 'home_ownership', \
  'initial_list_status', 'purpose', 'term', 'verification_status', 'sub_grade']
index_name = 'id'
label = 'loan_status' # 0 is DEFAULT, 1 is OK
datetime_name = 'issue_d'


#%%
# Split Data
# -------------

# Use data for from June and July for train and August for test:
train_df = data[data['issue_d'].dt.month.isin([6, 7])]
test_df = data[data['issue_d'].dt.month.isin([8])]


#%%
# Run Deepchecks for Train Test Validation
# ===========================================
#
# Define a Dataset Object
# -------------------------
#
# Create a deepchecks Dataset, including the relevant metadata (label, date, index, etc.).
# Check out :class:`deepchecks.tabular.Dataset` to see all of the columns that can be declared.

from deepchecks.tabular import Dataset

# We explicitly state the categorical features,
# otherwise they will be automatically inferred, which may not work perfectly and is not recommended.
# The label can be passed as a column name or a separate pd.Series / pd.DataFrame
# If existing, we should define also the index column and datetime column in the Dataset

train_ds = Dataset(train_df, label=label,cat_features=categorical_features, \
                   index_name=index_name, datetime_name=datetime_name)
test_ds = Dataset(test_df, label=label,cat_features=categorical_features, \
                   index_name=index_name, datetime_name=datetime_name)

#%%

# for convenience lets save it in a dictionary so we can reuse them for future Dataset initializations
columns_metadata = {'cat_features' : categorical_features, 'index_name': index_name,
                    'label':label, 'datetime_name':datetime_name}

#%%
# Run the Deepchecks Suite
# --------------------------
#
# Validate your data with the :class:`deepchecks.tabular.suites.train_test_validation` suite.
# It runs on two datasets, so you can use it to compare any two batches of data (e.g. train data, test data, a new batch of data
# that recently arrived)
#
# Check out the :doc:`when should you use </getting-started/when_should_you_use>`
# deepchecks guide for some more info about the existing suites and when to use them.

from deepchecks.tabular.suites import train_test_validation

validation_suite = train_test_validation()
suite_result = validation_suite.run(train_ds, test_ds)
suite_result
# Note: the result can be saved as html using suite_result.save_as_html()
# or exported to json using suite_result.to_json()

#%%
# We can see that we have a problem in the way we've split our data!
# We've mixed up data from two years, causing a potential leakage.
# Let's fix this.
# 
# Fix Data
# ^^^^^^^^^^

dt_col = data[datetime_name]
train_df = data[dt_col.dt.year.isin([2017]) & dt_col.dt.month.isin([6,7,8])]
test_df = data[dt_col.dt.year.isin([2018]) & dt_col.dt.month.isin([6,7,8])]

#%%

from deepchecks.tabular import Dataset

# Create the new Datasets
train_ds = Dataset(train_df, **columns_metadata)
test_ds = Dataset(test_df, **columns_metadata)

#%%
#
# Re-run Validation Suite
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#

suite_result = validation_suite.run(train_ds, test_ds)
suite_result

#%%
#
# Ok, the data leakage is fixed.
# However, in the current split We see we have a multivariate drift, detected by the WholeDatasetDrift check.
# The drift is detected mainly with a combination of the interest rate (``int_rate``) and loan grade (``sub_grade``).
# We can consider examining other sampling techniques (e.g. using only data from the same year), ideally achieving one in which the training data's
# univariate and multivariate distribution is similar to the data on which the model will run (test / production data).
# If we are planning on training a model with these splits, this drfit is worth investigating.
# Otherwise, we can 




#%%
# Run a Single Check
# -------------------
# We can run a single check on a dataset, and see the results.

# If we want to run only that check (possible with or without condition)
from deepchecks.tabular.checks import WholeDatasetDrift

check_with_condition = WholeDatasetDrift().add_condition_overall_drift_value_not_greater_than(0.4)
# check = WholeDatasetDrift()
dataset_drift_result = check_with_condition.run(train_ds, test_ds)

#%%
# We can also inspect and use the result's value:

dataset_drift_result.value
