# -*- coding: utf-8 -*-
"""
.. _plot_tabular_calibration_score:

Calibration Score
*****************
This notebook provides an overview for using and understanding the Calibration Score check.

**Structure:**

* `What is the Calibration Score check? <#what-is-the-calibration-score-check>`__
* `Binary Classification <#binary-classification>`__
* `Multi-class classification <#multi-class-classification>`__


What is the Calibration Score check?
==================================================
The ``CalibrationScore`` check calculates the calibration curve with brier score for each class.
Calibration curves (also known as reliability diagrams) compare how well the
probabilistic predictions of a binary classifier are calibrated. It plots the true
frequency of the positive label against its predicted probability, for binned predictions.
The Brier score metric may be used to assess how well a classifier is calibrated
(`Brier score <https://en.wikipedia.org/wiki/Brier_score>`_).

"""


#%%
# Imports
# =======

import warnings

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import CalibrationScore
from deepchecks.tabular.datasets.classification import adult


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning

#%%
# Binary Classification
# =====================

#%%
# Generate data & model
# --------------------

train_ds, test_ds = adult.load_data()
model = adult.load_fitted_model()

#%%
# Run the check
# --------------------

check = CalibrationScore()
check.run(test_ds, model)

#%%
# Multi-class classification
# ==========================


# Generate data & model
# --------------------
iris = load_iris(as_frame=True)
clf = LogisticRegression()
frame = iris.frame
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=55)
clf.fit(X_train, y_train)
ds = Dataset(pd.concat([X_test, y_test], axis=1), 
            features=iris.feature_names,
            label='target')

#%%
# Run the check
# --------------------
check = CalibrationScore()
check.run(ds, clf)
