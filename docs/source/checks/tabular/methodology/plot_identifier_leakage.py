# -*- coding: utf-8 -*-
"""
Identifier Leakage
******************
"""

#%%
# Imports
# =======

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.methodology import *

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
# Running ``identifier_leakage`` check
# ====================================

IdentifierLeakage().run(dataset)

#%%
# Using the ``SingleFeatureContribution`` check class
# ===================================================

my_check = IdentifierLeakage(ppscore_params={'sample': 10})
my_check.run(dataset=dataset)
