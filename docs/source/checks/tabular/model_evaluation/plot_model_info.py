# -*- coding: utf-8 -*-
"""
.. _plot_tabular_model_info:

Model Info
**********
"""

#%%

from sklearn.ensemble import AdaBoostClassifier

from deepchecks.tabular.checks import ModelInfo

#%%
clf = AdaBoostClassifier(learning_rate=1.2)
result = ModelInfo().run(clf)
