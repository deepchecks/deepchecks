# -*- coding: utf-8 -*-
"""
Segment Performance
*******************
"""

#%%
# Load data
# =========
# The dataset is the adult dataset which can be downloaded from the UCI machine learning repository.
#
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
# Irvine, CA: University of California, School of Information and Computer Science.

from sklearn.preprocessing import LabelEncoder
from deepchecks.tabular.datasets.classification import adult

label_name = 'income'

#%%
# Create Dataset
# ==============

train_ds, validation_ds = adult.load_data()
encoder = LabelEncoder()
train_ds.data[label_name] = encoder.fit_transform(train_ds.data[label_name])
validation_ds.data[label_name] = encoder.transform(validation_ds.data[label_name])


#%%
# Classification Model
# ====================

model = adult.load_fitted_model()
model

#%%

from deepchecks.tabular.checks.performance import SegmentPerformance

SegmentPerformance(feature_1='workclass', feature_2='hours-per-week').run(validation_ds, model)
