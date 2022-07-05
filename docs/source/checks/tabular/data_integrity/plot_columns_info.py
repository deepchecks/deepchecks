# -*- coding: utf-8 -*-
"""
.. _plot_tabular_columns_info:

Columns Info
************
Imports
=======
"""

#%%

import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ColumnsInfo

#%%
# Generating data
# ===============

num_fe = np.random.rand(500)
cat_fe = np.random.randint(3, size=500)
num_col = np.random.rand(500)
date = range(1635693229, 1635693729)
index = range(500)
data = {'index': index, 'date': date, 'a': cat_fe, 'b': num_fe, 'c': num_col, 'label': cat_fe}
df = pd.DataFrame.from_dict(data)

dataset = Dataset(df, label='label', datetime_name='date', index_name='index', features=['a', 'b'], cat_features=['a'])

#%%
# Running columns_info check
# ==========================

check = ColumnsInfo()

#%%

result = check.run(dataset=dataset)
