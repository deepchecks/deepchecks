# -*- coding: utf-8 -*-
"""
ROC Report
**********
"""

#%%
# Imports
# =======

from deepchecks.tabular import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from deepchecks.tabular.checks.performance import RocReport
import pandas as pd
import warnings

def custom_formatwarning(msg, *args, **kwargs):
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning

#%%
# Generating data
# ===============

iris = load_iris(as_frame=True)
clf = LogisticRegression(penalty='none')
frame = iris.frame
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=55)
clf.fit(X_train, y_train)
ds = Dataset(pd.concat([X_test, y_test], axis=1), 
            features=iris.feature_names,

#%%
# Running ``roc_report`` Check
# ============================

check = RocReport()

#%%

check.run(ds, clf)
