# -*- coding: utf-8 -*-
"""
Train Test Label Drift
**********************
"""

#%%

import pprint

import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestLabelDrift

#%%
# Generate data - Classification label
# ====================================

np.random.seed(42)

train_data = np.concatenate([np.random.randn(1000,2), np.random.choice(a=[1,0], p=[0.5, 0.5], size=(1000, 1))], axis=1)
#Create test_data with drift in label:
test_data = np.concatenate([np.random.randn(1000,2), np.random.choice(a=[1,0], p=[0.35, 0.65], size=(1000, 1))], axis=1) 

df_train = pd.DataFrame(train_data, columns=['col1', 'col2', 'target'])
df_test = pd.DataFrame(test_data, columns=['col1', 'col2', 'target'])

train_dataset = Dataset(df_train, label='target')
test_dataset = Dataset(df_test, label='target')

#%%

df_train.head()

#%%
# Run Check
# =========

check = TrainTestLabelDrift()
result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
result

#%%
# Generate data - Regression label
# ================================

train_data = np.concatenate([np.random.randn(1000,2), np.random.randn(1000, 1)], axis=1)
test_data = np.concatenate([np.random.randn(1000,2), np.random.randn(1000, 1)], axis=1)

df_train = pd.DataFrame(train_data, columns=['col1', 'col2', 'target'])
df_test = pd.DataFrame(test_data, columns=['col1', 'col2', 'target'])
#Create drift in test:
df_test['target'] = df_test['target'].astype('float') + abs(np.random.randn(1000)) + np.arange(0, 1, 0.001) * 4

train_dataset = Dataset(df_train, label='target')
test_dataset = Dataset(df_test, label='target')

#%%
# Run check
# =========

check = TrainTestLabelDrift()
result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
result

#%%
# Add condition

check_cond = TrainTestLabelDrift().add_condition_drift_score_not_greater_than()
check_cond.run(train_dataset=train_dataset, test_dataset=test_dataset)
