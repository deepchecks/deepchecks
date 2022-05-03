# -*- coding: utf-8 -*-
"""
Calibration Score
*****************
"""
#%%
# 

import warnings

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import CalibrationScore
from deepchecks.tabular.datasets.classification import adult


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning

#%%
# Binary Classification
# =====================
# Load data
# ---------
# The dataset is the adult dataset which can be downloaded from the UCI machine
# learning repository.
#
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
# Irvine, CA: University of California, School of Information and Computer Science.

from urllib.request import urlopen

from sklearn.preprocessing import LabelEncoder

label_name = 'income'

#%%

from deepchecks.tabular import Dataset

train_ds, test_ds = adult.load_data()

#%%

model = adult.load_fitted_model()
#%%

check = CalibrationScore()
check.run(test_ds, model)

#%%
# Multi-class classification
# ==========================

iris = load_iris(as_frame=True)
clf = LogisticRegression()
frame = iris.frame
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=55)
clf.fit(X_train, y_train)
ds = Dataset(pd.concat([X_test, y_test], axis=1), 
            features=iris.feature_names,
            label='target')

#%%

check = CalibrationScore()
check.run(ds, clf)
