# -*- coding: utf-8 -*-
"""
Model Info
**********
"""

#%%

from sklearn.ensemble import AdaBoostClassifier
from deepchecks.tabular.checks.overview import ModelInfo

#%%
clf = AdaBoostClassifier(learning_rate=1.2)
ModelInfo().run(clf)
