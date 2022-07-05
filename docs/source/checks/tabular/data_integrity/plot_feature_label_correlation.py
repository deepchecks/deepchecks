# -*- coding: utf-8 -*-
"""
.. _plot_tabular_feature_label_correlation:

Feature Label Correlation
***************************
"""

#%%
# Imports
# =======

import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation

#%%
# Generating Data
# ===============

df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
df['x4'] = df['x1'] * 0.5 + df['x2']
df['label'] = df['x2'] + 0.1 * df['x1']
df['x5'] = df['label'].apply(lambda x: 'v1' if x < 0 else 'v2')

#%%

ds = Dataset(df, label='label', cat_features=[])

#%%
# Using the FeatureLabelCorrelation check class
# ===============================================

my_check = FeatureLabelCorrelation(ppscore_params={'sample': 10})
result = my_check.run(dataset=ds)
