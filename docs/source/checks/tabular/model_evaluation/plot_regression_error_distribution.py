# -*- coding: utf-8 -*-
"""
Regression Error Distribution
*****************************
"""

#%%
# Imports
# =======

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import RegressionErrorDistribution

#%%
# Generating data
# ===============

diabetes_df = load_diabetes(return_X_y=False, as_frame=True).frame
train_df, test_df = train_test_split(diabetes_df, test_size=0.33, random_state=42)

train = Dataset(train_df, label='target', cat_features=['sex'])
test = Dataset(test_df, label='target', cat_features=['sex'])

clf = GradientBoostingRegressor(random_state=0)
_ = clf.fit(train.data[train.features], train.data[train.label_name])

#%%
# Running RegressionErrorDistribution check (normal distribution)
# ===============================================================

check = RegressionErrorDistribution()

#%%

check.run(test, clf)

#%%
# Skewing the data
# ----------------

test.data[test.label_name] = 150

#%%
# Running RegressionErrorDistribution check (abnormal distribution)
# =================================================================

check = RegressionErrorDistribution()

#%%

check.run(test, clf)
