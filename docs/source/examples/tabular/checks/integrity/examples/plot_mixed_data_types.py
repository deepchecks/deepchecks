# -*- coding: utf-8 -*-
"""
Mixed Data Types
****************
"""

#%%

from deepchecks.tabular.checks import MixedDataTypes
import pandas as pd

#%%
# Benign condition
# ================

data = {'col1': ['foo', 'bar', 'cat']}
dataframe = pd.DataFrame(data=data)
MixedDataTypes().add_condition_rare_type_ratio_not_in_range().run(dataframe)

#%%
# Issue detected
# ==============

data = {'col1': ['str', '1.0', 1, 2 , 2.61 , 't', 1, 1, 1,1,1], 'col2':['', '', '1.0', 'a', 'b', 'c', 'a', 'a', 'a', 'a','a'],
        'col3': [1,2,3,4,5,6,7,8, 9,10,11], 'col4': [1,2,3,4,5, 6, 7,8,'a',10,12]}
dataframe = pd.DataFrame(data=data)
MixedDataTypes().add_condition_rare_type_ratio_not_in_range().run(dataframe)
