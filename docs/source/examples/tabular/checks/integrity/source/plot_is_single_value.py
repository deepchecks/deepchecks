# -*- coding: utf-8 -*-
"""
Is Single Value
***************
"""

#%%
# Imports
# =======

from sklearn.datasets import load_iris
import pandas as pd
from deepchecks.tabular.checks.integrity.is_single_value import IsSingleValue

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

IsSingleValue().run(pd.DataFrame(X))

#%%

IsSingleValue().run(pd.DataFrame({'a':[3,4], 'b':[2,2], 'c':[None, None], 'd':['a', 4]}))

#%%

sv = IsSingleValue()
sv.run(df)

#%%

sv_ignore = IsSingleValue(ignore_columns=['b','c'])
sv_ignore.run(df)
