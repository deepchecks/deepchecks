# -*- coding: utf-8 -*-
"""
.. _plot_tabular_feature_label_correlation:

Feature Label Correlation
***************************

This notebook provides an overview for using and understanding the Feature Label Correlation check.

**Structure:**

* `What is Feature Label Correlation <#what-is-feature-label-correlation>`__
* `Generate data <#generate-data>`__
* `Run the check <#run-the-check>`__



What is Feature Label Correlation
==================================
The ``FeatureLabelCorrelation`` check computes the correlation between each
feature and the label, potentially spotting features highly correlated with the label.

This check works with 2 types of columns: categorical and numerical, and uses a different method to calculate the
correlation for each feature label pair:

1. numerical-numerical: `Pearson's correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__
2. numerical-categorical: `Correlation ratio <https://en.wikipedia.org/wiki/Correlation_ratio>`__
3. categorical-categorical: `Symmetric Theil's U <https://en.wikipedia.org/wiki/Uncertainty_coefficient>`__
"""

#%%
# Imports
# =======

import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation

#%%
# Generate Data
# ===============

df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
df['x4'] = df['x1'] * 0.5 + df['x2']
df['label'] = df['x2'] + 0.1 * df['x1']
df['x5'] = df['label'].apply(lambda x: 'v1' if x < 0 else 'v2')

#%%

ds = Dataset(df, label='label', cat_features=[])

#%%
# Run the check
# =================

my_check = FeatureLabelCorrelation(ppscore_params={'sample': 10})
my_check.run(dataset=ds)
