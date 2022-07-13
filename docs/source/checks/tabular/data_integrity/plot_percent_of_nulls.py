# -*- coding: utf-8 -*-
"""
.. _plot_percent_of_nulls:

Percent Of Nulls
****************

This page provides an overview for using the "Percent Of Nulls" check.

**Structure:**

* `Check Description <#check-description>`__
* `Run check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__


Check Description
=================

'Percent Of Nulls' check calculates percent of 'null' values for each column 
of provided to it dataset and displays result as a bar chart.
"""
#%%
# Run the Check
# =========
import numpy as np
import pandas as pd
from deepchecks.tabular.checks.data_integrity import PercentOfNulls

df = pd.DataFrame({'foo': [1, 2, None, np.nan], 'bar': [None, 1, 2, 3]})
result = PercentOfNulls().run(df)
result.show()

#%%
# Define a Condition
# ========================
df = pd.DataFrame({'foo': [1, 2, None, np.nan], 'bar': [None, 1, 2, 3]})
check = PercentOfNulls().add_condition_percent_of_nulls_not_greater_than()
result = check.run(df)
result.show()
