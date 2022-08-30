# -*- coding: utf-8 -*-
"""
.. _plot_percent_of_nulls:

Percent Of Nulls
****************

This notebook provides an overview for using the Percent Of Nulls check.

**Structure:**

* `What is Percent Of Nulls <#what-are-percent-of-nulls>`__
* `Generate data <#generate-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__


What is Percent Of Nulls
===========================

The ``PercentOfNulls`` check calculates percent of ``null`` values for each column
and displays the result as a bar chart.
"""

#%%
# Generate data
# ===============
import numpy as np
import pandas as pd
from deepchecks.tabular.checks.data_integrity import PercentOfNulls

df = pd.DataFrame({'foo': [1, 2, None, np.nan], 'bar': [None, 1, 2, 3]})

#%%
# Run the Check
# ================
result = PercentOfNulls().run(df)
result.show()

#%%
# Define a Condition
# =====================
df = pd.DataFrame({'foo': [1, 2, None, np.nan], 'bar': [None, 1, 2, 3]})
check = PercentOfNulls().add_condition_percent_of_nulls_not_greater_than()
result = check.run(df)
result.show()
