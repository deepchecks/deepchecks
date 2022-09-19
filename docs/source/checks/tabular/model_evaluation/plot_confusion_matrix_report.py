# -*- coding: utf-8 -*-
"""
.. _plot_tabular_confusion_matrix_report:

Confusion Matrix Report
***********************
This notebook provides an overview for using and understanding the Confusion Matrix Report check.


**Structure:**

* `What is the Confusion Matrix Report? <#what-is-the-confusion-matrix-report>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__


What is the Confusion Matrix Report?
======================================
The ``ConfusionMatrixReport`` produces a confusion matrix visualization which summarizes the
performance of the model. The confusion matrix contains the TP (true positive), FP (false positive),
TN (true negative) and FN (false negative), from which we can derive the relevant metrics,
such as accuracy, precision, recall etc. (`confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`__).
"""

#%%
# Imports
# =========

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ConfusionMatrixReport

#%%
# Generate data & model
# =======================

iris = load_iris(as_frame=True)
clf = AdaBoostClassifier()
frame = iris.frame
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
clf.fit(X_train, y_train)
ds = Dataset(pd.concat([X_test, y_test], axis=1), 
            features=iris.feature_names,
            label='target')

#%%
# Run the check
# ===============

check = ConfusionMatrixReport()
check.run(ds, clf)
