# -*- coding: utf-8 -*-
"""
.. _plot_tabular_is_single_value:

Is Single Value
***************
"""

#%%
# Imports
# =======

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from deepchecks.tabular.checks import IsSingleValue

#%%
# Load Data
# =========

iris = load_iris()
X = iris.data
df = pd.DataFrame({'a':[3,4,1], 'b':[2,2,2], 'c':[None, None, None], 'd':['a', 4, 6]})
df

#%%
# See functionality
# =================

result = IsSingleValue().run(pd.DataFrame(X))

#%%

result = IsSingleValue().run(pd.DataFrame({'a':[3,4], 'b':[2,2], 'c':[None, None], 'd':['a', 4]}))

#%%

sv = IsSingleValue()
result = sv.run(df)

#%%

sv_ignore = IsSingleValue(ignore_columns=['b','c'])
result = sv_ignore.run(df)

#%%

# Ignoring NaN values

result = IsSingleValue(ignore_nan=True).run(pd.DataFrame({
        'a': [3, np.nan],
        'b': [2, 2],
        'c': [None, np.nan],
        'd': ['a', 4]
    }))
