# -*- coding: utf-8 -*-
"""
String Mismatch Comparison
**************************
"""
#%%

from deepchecks.tabular.checks import StringMismatchComparison
import pandas as pd

#%%

data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}

StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data))
