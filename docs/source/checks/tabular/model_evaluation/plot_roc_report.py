# -*- coding: utf-8 -*-
"""
.. _plot_tabular_roc_report:

ROC Report
**********
This notebook provides an overview for using and understanding the ROC Report check.


**Structure:**

* `What is the ROC Report check? <#what-is-the-roc-report-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__


What is the ROC Report check?
==============================
The ``ROCReport`` check calculates the ROC curve for each class.
The ROC curve is a plot of TPR (true positive rate) with respect to FPR (false positive rate)
at various thresholds (`ROC curve <https://deepchecks.com/glossary/roc-receiver-operating-characteristic-curve>`__).
"""

#%%
# Imports
# ========

import warnings

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import RocReport


def custom_formatwarning(msg, *args, **kwargs):
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning

#%%
# Generate data & model
# ======================

iris = load_iris(as_frame=True)
clf = LogisticRegression(penalty='none')
frame = iris.frame
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=55)
clf.fit(X_train, y_train)
ds = Dataset(pd.concat([X_test, y_test], axis=1), 
            features=iris.feature_names,
            label='target')

#%%
# Run the Check
# ================

check = RocReport()
check.run(ds, clf)


#%%
# Define a condition
# ===================
# A condition for minimum allowed AUC score per class can be defined.
# Here, we define minimum AUC score to be 0.7.

check = RocReport()
check.add_condition_auc_greater_than(0.7).run(ds, clf)
