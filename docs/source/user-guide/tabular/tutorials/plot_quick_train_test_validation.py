# -*- coding: utf-8 -*-
"""
Train Test Validation Scenario on Lending Club Data - Quickstart
*********************************************************************

The deepchecks train test validation suite is relevant any time you wish to 
validate two data subsets. For example:

- Comparing distributions across different train-test splits (e.g. before 
  training a model or when splitting data for cross-validation)
- Comparing a new data batch to previous data batches

Here we'll use a loan's dataset 
(:mod:`deepchecks.tabular.datasets.classification.lending_club`),
to demonstrate how you can run the suite with only a few simple lines of code, 
and see which kind of insights it can find.

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


from deepchecks.tabular.datasets.classification import lending_club
import pandas as pd

data = lending_club.load_data(data_format='Dataframe', as_train_test=False)
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

# convert date column to datetime
data[datetime_name] = pd.to_datetime(data[datetime_name])

# Use data from June and July for train and August for test:
train_df = data[data[datetime_name].dt.month.isin([6, 7])]
test_df = data[data[datetime_name].dt.month.isin([8])]


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
# otherwise they will be automatically inferred,
# which may not work perfectly and is not recommended.

# The label can be passed as a column name or
# as a separate pd.Series / pd.DataFrame

# all metadata attributes are optional.
# Some checks require specific attributes and otherwise will not run.

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
# Validate your data with the :func:`deepchecks.tabular.suites.train_test_validation` suite.
# It runs on two datasets, so you can use it to compare any two batches of data (e.g. train data, test data, a new batch of data
# that recently arrived)
#
# Check out the :doc:`when should you use </getting-started/when_should_you_use>`
# deepchecks guide for some more info about the existing suites and when to use them.

from deepchecks.tabular.suites import train_test_validation

validation_suite = train_test_validation()
suite_result = validation_suite.run(train_ds, test_ds)
# Note: the result can be saved as html using suite_result.save_as_html()
# or exported to json using suite_result.to_json()
suite_result

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
# Ok, the date leakage doesn't exist anymore!
#
# However, we can see that we have a multivariate drift in the current split, 
# detected by the :doc:`</checks_gallery/tabular/train_test_validation/plot_whole_dataset_drift>` check.
# The drift is detected mainly by a combination of features representing the loan's interest rate (``int_rate``) 
# and its grade (``sub_grade``).
#
# We can consider examining other sampling techniques (e.g. using only data from the same year), 
# ideally achieving one in which the training data's
# univariate and multivariate distribution is 
# similar to the data on which the model will run (test / production data).
#
# If we are planning on training a model with these splits, this drift is worth understanding 
# (do we expect this kind of drift in the model's production environment? can we do something about it?).
# Otherwise, we can consider sampling or splitting the data differently, and using deepchecks to validate it.
# For more details about drift, see the :doc:`</user-guide/general/drift_guide>`.



#%%
# Run a Single Check
# -------------------
#
# We can run a single check on a dataset, and see the results.

# If we want to run only that check (possible with or without condition)
from deepchecks.tabular.checks import WholeDatasetDrift

check_with_condition = WholeDatasetDrift().add_condition_overall_drift_value_less_than(0.4)
# check = WholeDatasetDrift()
dataset_drift_result = check_with_condition.run(train_ds, test_ds)

#%%
# We can also inspect and use the result's value:

dataset_drift_result.value

#%%
# and see if the conditions have passed
dataset_drift_result.passed_conditions()

#%%
# Create a Custom Suite
# ----------------------
#
# To create our own suite, we can simply write all of the checks, and add optional conditions.

from deepchecks.tabular import Suite
from deepchecks.tabular.checks import TrainTestFeatureDrift, WholeDatasetDrift, \
 TrainTestPredictionDrift, TrainTestLabelDrift

drift_suite = Suite('drift suite',
TrainTestFeatureDrift().add_condition_drift_score_less_than(
  max_allowed_categorical_score=0.2, max_allowed_numeric_score=0.1),
WholeDatasetDrift().add_condition_overall_drift_value_less_than(0.4),
TrainTestLabelDrift(),
TrainTestPredictionDrift()
)

#%%
#
# we can run our new suite using:

result = drift_suite.run(train_ds, test_ds)
