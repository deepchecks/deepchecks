# -*- coding: utf-8 -*-
"""
Model Error Analysis
********************
"""

#%%
# Load Data
# =========
# The dataset is the adult dataset which can be downloaded from the UCI machine learning repository.
#
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
# Irvine, CA: University of California, School of Information and Computer Science.

import pandas as pd
from urllib.request import urlopen
from sklearn.preprocessing import LabelEncoder
from deepchecks.tabular.datasets.classification import adult

#%%
# Create Dataset
# ==============

label_name = 'income'
train_ds, test_ds = adult.load_data()
encoder = LabelEncoder()
train_ds.data[label_name] = encoder.fit_transform(train_ds.data[label_name])
test_ds.data[label_name] = encoder.transform(test_ds.data[label_name])

#%%
# Classification Model
# ====================

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

model = adult.load_fitted_model()

#%%
# Run Check
# =========

from deepchecks.tabular.checks import ModelErrorAnalysis

#%%

check = ModelErrorAnalysis(min_error_model_score=0.3)
check = check.add_condition_segments_performance_relative_difference_not_greater_than()
res = check.run(train_ds, test_ds, model)
res

#%%

res.value
