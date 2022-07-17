# -*- coding: utf-8 -*-
"""
.. _quick_data_integrity:

Quickstart - Data Integrity Suite
*********************************

The deepchecks integrity suite is relevant any time you have data that you wish to validate:
whether it's on a fresh batch of data, or right before splitting it or using it for training. 
Here we'll use the avocado prices dataset (:mod:`deepchecks.tabular.datasets.regression.avocado`),
to demonstrate how you can run the suite with only a few simple lines of code,
and see which kind of insights it can find.

.. code-block:: bash

    # Before we start, if you don't have deepchecks installed yet, run:
    import sys
    !{sys.executable} -m pip install deepchecks -U --quiet

    # or install using pip from your python environment
"""

#%%
# Load and Prepare Data
# ====================================================

from deepchecks.tabular import datasets

# load data
data = datasets.regression.avocado.load_data(data_format='DataFrame', as_train_test=False)

#%%
# Insert a few typcial problems to dataset for demonstration.

import pandas as pd

def add_dirty_data(df):
    # change strings
    df.loc[df[df['type'] == 'organic'].sample(frac=0.18).index,'type'] = 'Organic'
    df.loc[df[df['type'] == 'organic'].sample(frac=0.01).index,'type'] = 'ORGANIC'
    # add duplicates
    df = pd.concat([df, df.sample(frac=0.156)], axis=0, ignore_index=True)
    # add column with single value
    df['Is Ripe'] = True
    return df


dirty_df = add_dirty_data(data)

#%%
# Run Deepchecks for Data Integrity
# ====================================
#
# Create a Dataset Object
# ------------------------
#
# Create a deepchecks Dataset, including the relevant metadata (label, date, index, etc.).
# Check out :class:`deepchecks.tabular.Dataset` to see all of the columns and types 
# that can be declared.

from deepchecks.tabular import Dataset

# Categorical features can be heuristically inferred, however we
# recommend to state them explicitly to avoid misclassification.

# Metadata attributes are optional. Some checks will run only if specific attributes are declared.

ds = Dataset(dirty_df, cat_features= ['type'], datetime_name='Date', label= 'AveragePrice')

#%%
# Run the Deepchecks Suite
# --------------------------
#
# Validate your data with the :func:`deepchecks.tabular.suites.data_integrity` suite.
# It runs on a single dataset, so you can run it on any batch of data (e.g. train data, test data, a new batch of data
# that recently arrived)
#
# Check out the :doc:`when should you use </getting-started/when_should_you_use>`
# deepchecks guide for some more info about the existing suites and when to use them.

from deepchecks.tabular.suites import data_integrity

# Run Suite:
integ_suite = data_integrity()
suite_result = integ_suite.run(ds)
# Note: the result can be saved as html using suite_result.save_as_html()
# or exported to json using suite_result.to_json()
suite_result.show()

#%%
# We can inspect the suite outputs and see that there are a few problems we'd like to fix.
# We'll now fix them and check that they're resolved by re-running those specific checks.


#%%
# Run a Single Check
# -------------------
# We can run a single check on a dataset, and see the results.

from deepchecks.tabular.checks import IsSingleValue, DataDuplicates

# first let's see how the check runs:
IsSingleValue().run(ds)

#%%

# we can also add a condition:
single_value_with_condition = IsSingleValue().add_condition_not_single_value()
result = single_value_with_condition.run(ds)
result.show()

#%%

# We can also inspect and use the result's value:
result.value

#%%
# Now let's remove the single value column and rerun (notice that we're using directly 
# the ``data`` attribute that stores the dataframe inside the Dataset)

ds.data.drop('Is Ripe', axis=1, inplace=True)
result = single_value_with_condition.run(ds)
result.show()

#%%

# Alternatively we can fix the dataframe directly, and create a new dataset.
# Let's fix also the duplicate values:
dirty_df.drop_duplicates(inplace=True)
dirty_df.drop('Is Ripe', axis=1, inplace=True)
ds = Dataset(dirty_df, cat_features=['type'], datetime_name='Date', label='AveragePrice')
result = DataDuplicates().add_condition_ratio_less_or_equal(0).run(ds)
result.show()

#%%
# Rerun Suite on the Fixed Dataset
# ---------------------------------
# Finally, we'll choose to keep the "organic" multiple spellings as they represent different sources.
# So we'll customaize the suite by removing the condition from it (or delete check completely).
# Alternatively - we can customize it by creating a new Suite with the desired checks and conditions.
# See :doc:`/user-guide/general/customizations/examples/plot_create_a_custom_suite` for more info.

# let's inspect the suite's structure
integ_suite

#%%

# and remove the condition:
integ_suite[3].clean_conditions()

#%%
# Now we can re-run the suite using:
res = integ_suite.run(ds)

#%%
# and all of the conditions will pass.
#
# *Note: the check we manipulated will still run as part of the Suite, however
# it won't appear in the Conditions Summary since it no longer has any
# conditions defined on it. You can still see its display results in the 
# Additional Outputs section*
#
# For more info about working with conditions, see the detailed
# :doc:`/user-guide/general/customizations/examples/plot_configure_check_conditions` guide.
