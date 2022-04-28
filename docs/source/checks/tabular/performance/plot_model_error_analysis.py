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
from deepchecks.tabular.datasets.classification import adult

#%%
# Create Dataset
# ==============

train_ds, test_ds = adult.load_data()

#%%
# Classification Model
# ====================

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
