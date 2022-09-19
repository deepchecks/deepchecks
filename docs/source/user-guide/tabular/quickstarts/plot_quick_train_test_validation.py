# -*- coding: utf-8 -*-
"""
.. _quick_train_test_validation:

Quickstart - Train-Test Validation Suite
****************************************

The deepchecks train-test validation suite is relevant any time you wish to 
validate two data subsets. For example:

- Comparing distributions across different train-test splits (e.g. before 
  training a model or when splitting data for cross-validation)
- Comparing a new data batch to previous data batches

Here we'll use a loans' dataset
(:mod:`deepchecks.tabular.datasets.classification.lending_club`),
to demonstrate how you can run the suite with only a few simple lines of code, 
and see which kind of insights it can find.

.. code-block:: bash

    # Before we start, if you don't have deepchecks installed yet, run:
    import sys
    !{sys.executable} -m pip install deepchecks -U --quiet

    # or install using pip from your python environment
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
# Split Data to Train and Test
# -----------------------------

# convert date column to datetime, `issue_d`` is date column
data['issue_d'] = pd.to_datetime(data['issue_d'])

# Use data from June and July for train and August for test:
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
# Check out :class:`deepchecks.tabular.Dataset` to see all of the columns and types 
# that can be declared.


#%%
# Define Lending Club Metadata
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

categorical_features = ['addr_state', 'application_type', 'home_ownership', \
  'initial_list_status', 'purpose', 'term', 'verification_status', 'sub_grade']
index_name = 'id'
label = 'loan_status' # 0 is DEFAULT, 1 is OK
datetime_name = 'issue_d'


#%%
# Create Dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^

from deepchecks.tabular import Dataset

# Categorical features can be heuristically inferred, however we
# recommend to state them explicitly to avoid misclassification.

# Metadata attributes are optional. Some checks will run only if specific attributes are declared.

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
# Check out the :doc:`"when should you use deepchecks guide" </getting-started/when_should_you_use>`
# for some more info about the existing suites and when to use them.

from deepchecks.tabular.suites import train_test_validation

validation_suite = train_test_validation()
suite_result = validation_suite.run(train_ds, test_ds)
# Note: the result can be saved as html using suite_result.save_as_html()
# or exported to json using suite_result.to_json()
suite_result

#%%
# As you can see in the suite's results: the Date Train-Test Leakage check failed,
# indicating that we may have a problem in the way we've split our data!
# We've mixed up data from two years, causing a leakage of future data
# in the training dataset.
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
suite_result.show()

#%%
#
# Ok, the date leakage doesn't happen anymore!
#
# However, in the current split after the fix, we can see that we have a multivariate drift, 
# detected by the :doc:`/checks_gallery/tabular/train_test_validation/plot_multivariate_drift` check.
# The drift is caused mainly by a combination of features representing the loan's interest rate (``int_rate``) 
# and its grade (``sub_grade``). In order to proceed, we should think about the two options we have: 
# To split the data in a different manner, or to stay with the current split.
#
# For working with different data splits: We can consider examining other sampling techniques 
# (e.g. using only data from the same year), ideally achieving one in which the training data's
# univariate and multivariate distribution is similar to the data 
# on which the model will run (test / production data). 
# Of course, we can use deepchecks to validate the new splits.
#
# If the current split is representative and we are planning on training a model with it, 
# it is worth understanding this drift (do we expect this kind of drift in the model's 
# production environment? can we do something about it?).
#
# For more details about drift, see the :doc:`/user-guide/general/drift_guide`.



#%%
# Run a Single Check
# -------------------
#
# We can run a single check on a dataset, and see the results.

# If we want to run only that check (possible with or without condition)
from deepchecks.tabular.checks import MultivariateDrift

check_with_condition = MultivariateDrift().add_condition_overall_drift_value_less_than(0.4)
# or just the check without the condition:
# check = MultivariateDrift()
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
from deepchecks.tabular.checks import TrainTestFeatureDrift, MultivariateDrift, \
 TrainTestPredictionDrift, TrainTestLabelDrift

drift_suite = Suite('drift suite',
TrainTestFeatureDrift().add_condition_drift_score_less_than(
  max_allowed_categorical_score=0.2, max_allowed_numeric_score=0.1),
MultivariateDrift().add_condition_overall_drift_value_less_than(0.4),
TrainTestLabelDrift(),
TrainTestPredictionDrift()
)

#%%
#
# we can run our new suite using:

result = drift_suite.run(train_ds, test_ds)
result.show()
