# -*- coding: utf-8 -*-
"""
Identifier Label Correlation
******************
"""

#%%
# Imports
# =======

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import IdentifierLabelCorrelation

#%%
# Generating Data

np.random.seed(42)
df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
df['x4'] = df['x1'] * 0.05 + df['x2']
df['x5'] = df['x2']*121 + 0.01 * df['x1']
df['label'] = df['x5'].apply(lambda x: 0 if x < 0 else 1)

#%%

dataset = Dataset(df, label='label', index_name='x1', datetime_name='x2')

#%%
# Running ``IdentifierLabelCorrelation`` check
# ====================================

IdentifierLabelCorrelation().run(dataset)

#%%
# Using the ``IdentifierLabelCorrelation`` check class
# ===================================================

my_check = IdentifierLabelCorrelation(ppscore_params={'sample': 10})
my_check.run(dataset=dataset)
