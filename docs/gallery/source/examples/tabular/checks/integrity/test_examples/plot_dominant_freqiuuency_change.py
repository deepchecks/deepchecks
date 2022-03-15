# -*- coding: utf-8 -*-
"""
Dominant Frequency Change
*************************
"""

#%%
# Imports
# =======

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from deepchecks.tabular.checks.integrity import DominantFrequencyChange
from deepchecks.tabular.base import Dataset

#%%
# Generating Data
# ===============

iris = load_iris(return_X_y=False, as_frame=True)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)
train_dataset = Dataset(pd.concat([X_train, y_train], axis=1), 
            features=iris.feature_names,
            label='target')

test_df = pd.concat([X_test, y_test], axis=1)

# make duplicates in the test data
test_df.loc[test_df.index % 2 == 0, 'petal length (cm)'] = 5.1
test_df.loc[test_df.index / 3 > 8, 'sepal width (cm)'] = 2.7

validation_dataset = Dataset(test_df, 
            features=iris.feature_names,
            label='target')

#%%
# Running ``dominant_frequency_change`` check
# ===========================================

check = DominantFrequencyChange()
check.run(validation_dataset, train_dataset)
