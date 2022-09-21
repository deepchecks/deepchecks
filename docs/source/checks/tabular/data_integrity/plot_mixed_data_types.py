# -*- coding: utf-8 -*-
"""
.. _plot_tabular_mixed_data_types:

Mixed Data Types
****************

This notebook provides an overview for using and understanding the mixed data types check.

**Structure:**

* `What are Mixed Data Types? <#what-are-mixed-data-types>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

What are Mixed Data Types?
==========================
Mixed data types is when a column contains both string values and numeric values (either as numeric type or
as string like "42.90"). This may indicate a problem in the data collection pipeline, or represent a
problem situation for the model's training.

This checks searches for columns with a mix of strings and numeric values and returns them and their respective ratios.

Run the Check
=============
We will run the check on the adult dataset which can be downloaded from the
`UCI machine learning repository <http://archive.ics.uci.edu/ml>`_ and is also available in
`deepchecks.tabular.datasets`, and introduce to it some data type mixing in order to show the check's result.
"""

import numpy as np
import pandas as pd

from deepchecks.tabular.datasets.classification import adult

# Prepare functions to insert mixed data types

def insert_new_values_types(col: pd.Series, ratio_to_replace: float, values_list):
    col = col.to_numpy().astype(object)
    indices_to_replace = np.random.choice(range(len(col)), int(len(col) * ratio_to_replace), replace=False)
    new_values = np.random.choice(values_list, len(indices_to_replace))
    col[indices_to_replace] = new_values
    return col


def insert_string_types(col: pd.Series, ratio_to_replace):
    return insert_new_values_types(col, ratio_to_replace, ['a', 'b', 'c'])


def insert_numeric_string_types(col: pd.Series, ratio_to_replace):
    return insert_new_values_types(col, ratio_to_replace, ['1.0', '1', '10394.33'])


def insert_number_types(col: pd.Series, ratio_to_replace):
    return insert_new_values_types(col, ratio_to_replace, [66, 99.9])


# Load dataset and insert some data type mixing
adult_df, _ = adult.load_data(as_train_test=True, data_format='Dataframe')
adult_df['workclass'] = insert_numeric_string_types(adult_df['workclass'], ratio_to_replace=0.01)
adult_df['education'] = insert_number_types(adult_df['education'], ratio_to_replace=0.1)
adult_df['age'] = insert_string_types(adult_df['age'], ratio_to_replace=0.5)

#%%

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import MixedDataTypes

adult_dataset = Dataset(adult_df, cat_features=['workclass', 'education'])
check = MixedDataTypes()
result = check.run(adult_dataset)
result

#%%
# Define a Condition
# ==================
# We can define a condition that enforces the ratio of the "rare type" (the less common type, either numeric or string)
# is not in a given range. The range represents the dangerous zone, when the ratio is lower than the lower bound, then
# it's presumably a contamination but a negligible one, and when the ratio is higher than the upper bound, then it's
# presumably supposed to contain both numbers and string values. So when the ratio is inside the range there is a real
# chance that the rarer data type may represent a problem to model training and inference.

check = MixedDataTypes().add_condition_rare_type_ratio_not_in_range((0.01, 0.2))
result = check.run(adult_dataset)
result.show(show_additional_outputs=False)
