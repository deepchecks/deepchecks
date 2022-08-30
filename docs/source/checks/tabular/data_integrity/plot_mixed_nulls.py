# -*- coding: utf-8 -*-
"""
.. _plot_tabular_mixed_nulls:

Mixed Nulls
***********

This notebook provides an overview for using and understanding the Mixed Nulls check.

**Structure:**

* `What are Mixed Nulls <#what-are-mixed-nulls>`__
* `Generate data <#generate-data>`__
* `Run the check <#run-the-check>`__
"""

#%%
# What are Mixed Nulls
# ======================
# The ``MixedNulls`` check search for various types of null values, including string representations of null.


#%%
# Imports
# =======
import pandas as pd

from deepchecks.tabular.checks import MixedNulls

#%%
# Generate data
# ===============

data = {'col1': ['sog', '1', 'cat', None, None]}
dataframe = pd.DataFrame(data=data)


#%%
# Run the check
# ===============

MixedNulls().run(dataframe)

#%%
# We can also check for string representations of null:

data = {'col1': ['nan', None, 'null', 'Nan', '1', 'cat'], 'col2':['', '', 'None', 'a', 'b', 'c'], 'col3': [1,2,3,4,5,6]}
dataframe = pd.DataFrame(data=data)
MixedNulls().run(dataframe)
