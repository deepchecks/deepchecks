# -*- coding: utf-8 -*-
"""
Datasets Size Comparison
************************
"""

#%%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.methodology import DatasetsSizeComparison

#%%

df = pd.DataFrame(np.random.randn(1000, 3), columns=['x1', 'x2', 'x3'])
df['label'] = df['x2'] + 0.1 * df['x1']

train, test = train_test_split(df, test_size=0.4)
train = Dataset(train, features=['x1', 'x2', 'x3'], label='label')
test = Dataset(test, features=['x1', 'x2', 'x3'], label='label')

check_instance = (
    DatasetsSizeComparison()
    .add_condition_train_dataset_not_smaller_than_test()
    .add_condition_test_size_not_smaller_than(100)
    .add_condition_test_train_size_ratio_not_smaller_than(0.2)
)

#%%

check_instance.run(train, test)
