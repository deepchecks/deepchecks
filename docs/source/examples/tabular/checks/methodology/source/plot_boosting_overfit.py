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

from deepchecks.tabular.datasets.classification import adult

train_df, val_df = adult.load_data(data_format='Dataframe')

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

train_ds = Dataset(train_df, label='income')
validation_ds = Dataset(val_df, label='income')

#%%
# Classification model
# ====================

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(random_state=0)
clf.fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])
BoostingOverfit().run(train_ds, validation_ds, clf)
