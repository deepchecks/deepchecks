# -*- coding: utf-8 -*-
"""
Unused Features
***************
"""

#%%

from deepchecks.tabular import Dataset
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from deepchecks.tabular.checks import UnusedFeatures
import pandas as pd

#%%
# Load data
# =========
# The dataset is the adult dataset which can be downloaded from the UCI machine learning repository.
#
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
# Irvine, CA: University of California, School of Information and Computer Science.

from deepchecks.tabular.datasets.classification import adult

label_name = 'income'


#%%
# Create datasets
# ===============

train_ds, test_ds = adult.load_data()

#%%
# Classification model
# ====================

model = adult.load_fitted_model()

#%%

UnusedFeatures().run(train_ds, test_ds, model)

#%%
# **stricter threshold on what features are considered "high variance"**

UnusedFeatures(feature_variance_threshold=1.5).run(train_ds, test_ds, model)

#%%
# **All features are considered important**

UnusedFeatures(feature_importance_threshold=0).run(train_ds, test_ds, model)
