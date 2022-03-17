# -*- coding: utf-8 -*-
"""
Mixed Nulls
***********
"""

#%%

from deepchecks.tabular.checks import MixedNulls
import pandas as pd

#%%

data = {'col1': ['sog', '1', 'cat', None, None]}
dataframe = pd.DataFrame(data=data)
MixedNulls().run(dataframe)

#%%

data = {'col1': ['nan', None, 'null', 'Nan', '1', 'cat'], 'col2':['', '', 'None', 'a', 'b', 'c'], 'col3': [1,2,3,4,5,6]}
dataframe = pd.DataFrame(data=data)
MixedNulls().run(dataframe)
