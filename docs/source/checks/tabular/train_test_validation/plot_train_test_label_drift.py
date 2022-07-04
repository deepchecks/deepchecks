# -*- coding: utf-8 -*-
"""
.. _plot_tabular_train_test_label_drift:

Train Test Label Drift
**********************

This notebooks provides an overview for using and understanding label drift check.

**Structure:**

* `What Is Label Drift? <#what-is-label-drift>`__
* `Run Check on a Classification Label <#run-check-on-a-classification-label>`__
* `Run Check on a Regression Label <#run-check-on-a-regression-label>`__
* `Add a Condition <#run-check>`__

What Is Label Drift?
========================
Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Label drift is when drift occurs in the label itself.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide>`.

How Deepchecks Detects Label Drift
------------------------------------

This check detects label drift by using :ref:`univariate measures <drift_detection_by_univariate_measure>`
on the label column.

"""

#%%

import pprint

import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestLabelDrift

#%%
# Run Check on a Classification Label
# ====================================

# Generate data:
# --------------

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
# ===============================

check = TrainTestLabelDrift()
result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
result

#%%
# Run Check on a Regression Label
# ================================

# Generate data:
# --------------

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
# ---------

check = TrainTestLabelDrift()
result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
result

#%%
# Add a Condition
# ===============

check_cond = TrainTestLabelDrift().add_condition_drift_score_less_than()
check_cond.run(train_dataset=train_dataset, test_dataset=test_dataset)
