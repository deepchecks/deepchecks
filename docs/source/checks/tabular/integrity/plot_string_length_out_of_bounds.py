# -*- coding: utf-8 -*-
"""
String Length Out Of Bounds
***************************
"""

#%%

import pandas as pd

from deepchecks.tabular.checks import StringLengthOutOfBounds

#%%

col1 = ["aaaaa33", "aaaaaaa33"]*40
col1.append("a")
col1.append("aaaaaadsfasdfasdf")

col2 = ["b", "abc"]*41

col3 = ["a"]*80
col3.append("a"*100)
col3.append("a"*200)
# col1 and col3 contrains outliers, col2 does not
df = pd.DataFrame({"col1":col1, "col2": col2, "col3": col3 })

#%%

StringLengthOutOfBounds(min_unique_value_ratio=0.01).run(df)

#%%

col = ["a","a","a","a","a","a","a","a","a","a","a","a","a","ab","ab","ab","ab","ab","ab", "ab"]*1000
col.append("basdbadsbaaaaaaaaaa")
col.append("basdbadsbaaaaaaaaaaa")
df = pd.DataFrame({"col1":col})
StringLengthOutOfBounds(num_percentiles=1000, min_unique_values=3).run(df)
