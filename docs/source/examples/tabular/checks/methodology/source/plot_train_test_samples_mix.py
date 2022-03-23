# -*- coding: utf-8 -*-
"""
Train Test Samples Mix
**********************
"""
#%%
# Imports
# =======

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from deepchecks.tabular.checks.methodology import TrainTestSamplesMix
from deepchecks.tabular import Dataset

#%%
# Generating data
# ===============

iris = load_iris(return_X_y=False, as_frame=True)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)
train_dataset = Dataset(pd.concat([X_train, y_train], axis=1), 
            features=iris.feature_names,
            label='target')

test_df = pd.concat([X_test, y_test], axis=1)
bad_test = test_df.append(train_dataset.data.iloc[[0, 1, 1, 2, 3, 4]], ignore_index=True)
                    
test_dataset = Dataset(bad_test, 
            features=iris.feature_names,
            label='target')

#%%
# Running ``data_sample_leakage_report check``
# ============================================

check = TrainTestSamplesMix()
check.run(test_dataset=test_dataset, train_dataset=train_dataset)
