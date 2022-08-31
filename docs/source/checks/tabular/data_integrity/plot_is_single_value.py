# -*- coding: utf-8 -*-
"""
.. _plot_tabular_is_single_value:

Is Single Value
***************

This notebook provides an overview for using and understanding the Is Single Value check.

**Structure:**

* `What is the Is Single Value check <#what-is-feature-label-correlation>`__
* `Load Data <#load-data>`__
* `Run the check <#run-the-check>`__
"""

# %%
# What is the Is Single Value check
# ====================================
# The ``IsSingleValue`` check checks if there are columns which have only a single unique
# value in all rows.


# %%
# Imports
# =======

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from deepchecks.tabular.checks import IsSingleValue

# %%
# Load Data
# =========

iris = load_iris()
X = iris.data

# %%
# Run the check
# =================

IsSingleValue().run(pd.DataFrame(X))


# %%
# If ``None`` is given as a value, it will be ignored (this can be changed with ``ignore_nan`` set to ``False``):
df = pd.DataFrame({'a': [3, 4, 1], 'b': [2, 2, 2], 'c': [None, None, None], 'd': ['a', 4, 6]})
sv = IsSingleValue()
sv.run(df)


# %%

# Ignoring NaN values:
IsSingleValue(ignore_nan=True).run(pd.DataFrame({
    'a': [3, np.nan],
    'b': [2, 2],
    'c': [None, np.nan],
    'd': ['a', 4]
}))

# %%
# Ignoring specific columns by name is also possible:
sv_ignore = IsSingleValue(ignore_columns=['b', 'c'])
sv_ignore.run(df)
