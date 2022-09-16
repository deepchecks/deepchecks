# -*- coding: utf-8 -*-
"""
.. _plot_tabular_model_info:

Model Info
**********

This notebook provides an overview for using and understanding the Model Info check.

**Structure:**

* `What is a Model Info? <#what-is-the-model-info-check>`__
* `Run the check <#run-the-check>`__


What is the Model Info check?
===========================
The ``ModelInfo`` check produce a summary for all model's parameters (number of estimators,
learning rate, verbosity etc.).

"""

#%%
# Imports
# =============
from sklearn.ensemble import AdaBoostClassifier

from deepchecks.tabular.checks import ModelInfo

#%%
# Run the check
# =============
clf = AdaBoostClassifier(learning_rate=1.2)
ModelInfo().run(clf)
