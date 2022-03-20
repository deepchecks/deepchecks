# -*- coding: utf-8 -*-
"""
Boosting Overfit
****************
Load Data
=========
The dataset is the adult dataset which can be downloaded from the UCI machine
learning repository.

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.
"""

#%%

import pandas as pd
from sklearn.preprocessing import LabelEncoder

names = [*(f'col_{i}' for i in range(1,14)), 'target']
train_df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 
                       header=None, names=names)
val_df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', 
                     skiprows=1, header=None, names=names)
val_df['target'] = val_df['target'].str[:-1]

# Run label encoder on all categorical columns
for column in train_df.columns:
    if train_df[column].dtype == 'object':
        le = LabelEncoder()
        le.fit(pd.concat([train_df[column], val_df[column]]))
        train_df[column] = le.transform(train_df[column])
        val_df[column] = le.transform(val_df[column])

#%%
# Create Dataset
# ==============

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.methodology.boosting_overfit import BoostingOverfit

train_ds = Dataset(train_df, label='target')
validation_ds = Dataset(val_df, label='target')

#%%
# Classification model
# ====================

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(random_state=0)
clf.fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])
BoostingOverfit().run(train_ds, validation_ds, clf)
