# -*- coding: utf-8 -*-
"""
.. _plot_tabular_model_info:

Model Info
***********

This notebook provides an overview for using and understanding the Model Info check.

**Structure:**

* `What is the Model Info check? <#what-is-the-model-info-check>`__
* `Run the check <#run-the-check>`__


What is the Model Info check?
<<<<<<< HEAD
===============================
=======
===========================
>>>>>>> f4b3669b28960b7b25a04699cd17b84983a0be0d
The ``ModelInfo`` check produces a summary for the model parameters (number of estimators,
learning rate, verbosity etc.).

"""

#%%
# Imports
# =============
from sklearn.ensemble import AdaBoostClassifier

from deepchecks.tabular.checks import ModelInfo

#%%
# Run the check
# ===============
clf = AdaBoostClassifier(learning_rate=1.2)
ModelInfo().run(clf)
