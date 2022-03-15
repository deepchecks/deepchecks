# -*- coding: utf-8 -*-
"""
String Mismatch
***************
"""

#%%

from deepchecks.tabular.checks import StringMismatch
import pandas as pd

data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'foo?']}
df = pd.DataFrame(data=data)
r = StringMismatch().run(df)

#%%

c  = StringMismatch().add_condition_no_variants()
c.conditions_decision(r)
