# -*- coding: utf-8 -*-
"""
Performance Report
******************
"""

#%%
# Imports
# =======

from deepchecks.tabular import Dataset
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from deepchecks.tabular.checks.performance import PerformanceReport

#%%
# Generating data
# ===============

iris = load_iris(as_frame=True)
clf = AdaBoostClassifier()
frame = iris.frame
X = iris.data
Y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.33, random_state=42)
train_ds = Dataset(pd.concat([X_train, y_train], axis=1), 
            features=iris.feature_names,
            label='target')
test_ds = Dataset(pd.concat([X_test, y_test], axis=1), 
            features=iris.feature_names,
            label='target')
_ = clf.fit(X_train, y_train)

#%%
# Running peformance report on classification
# ===========================================

check = PerformanceReport()
check.run(train_ds, test_ds, clf)

#%%
# Generate regression data
# ========================

from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
clf = AdaBoostRegressor()

frame = diabetes.frame
X = diabetes.data
Y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.33, random_state=42)
train_ds = Dataset(pd.concat([X_train, y_train], axis=1), 
            features=diabetes.feature_names,
            label='target', cat_features=['sex'])
test_ds = Dataset(pd.concat([X_test, y_test], axis=1), 
            features=diabetes.feature_names,
            label='target', cat_features=['sex'])
_ = clf.fit(X_train, y_train)

#%%
# Run performance report on regression
# ====================================

check = PerformanceReport()
check.run(train_ds, test_ds, clf)
