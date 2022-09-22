# -*- coding: utf-8 -*-
"""
.. _plot_tabular_boosting_overfit:

Boosting Overfit
****************

This notebook provides an overview for using and understanding the boosting overfit check.

**Structure:**

* `What is a boosting overfit? <#what-is-a-boosting-overfit>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is A Boosting Overfit?
===========================
A boosting algorithm is a machine learning algorithm that uses a combination of weak learners to predict a target
variable. The mechanism of boosting is to increase the number of weak learners in the ensemble by iteratively adding a
new weak learner. The new weak learner uses the error of the ensemble from the previous iterations as its training data.
This mechanism continues until the ensemble reaches a certain performance level or until the given maximum number of
iterations is reached.

Thanks to its mechanism, boosting algorithms are usually less prone to overfitting than other traditional algorithms
like single decision trees. However, the number of weak learners in the ensemble can be too large making the ensemble
too complex given the amount of data it was trained on. In this case, the ensemble may be overfitted on the training
data.

How deepchecks detects a boosting overfit?
------------------------------------------
The check runs for a pre-defined number of iterations, and in each step it uses only the first X estimators from the
boosting model when predicting the target variable (number of estimators X is monotonic increasing).
It plots the given score calculated for each iteration for both the train dataset and the test dataset.

If the ratio of decline between the maximal test score achieved in any boosting iteration and the test score achieved in the
last iteration ("full" model score) is above a given threshold (0.05 by default), it means the model is overfitted
and the default condition, if added, will fail.

Supported Models
----------------
Currently the check supports the following models:

*   AdaBoost (sklearn)
*   GradientBoosting (sklearn)
*   XGBoost (xgboost)
*   LGBM (lightgbm)
*   CatBoost (catboost)

Generate data & model
=====================

The dataset is the adult dataset which can be downloaded from the UCI machine
learning repository.

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from deepchecks.tabular import Dataset
from deepchecks.tabular.datasets.classification import adult

train_df, val_df = adult.load_data(data_format='Dataframe')

# Run label encoder on all categorical columns
for column in train_df.columns:
    if train_df[column].dtype == 'object':
        le = LabelEncoder()
        le.fit(pd.concat([train_df[column], val_df[column]]))
        train_df[column] = le.transform(train_df[column])
        val_df[column] = le.transform(val_df[column])

train_ds = Dataset(train_df, label='income')
validation_ds = Dataset(val_df, label='income')

#%%
# Classification model
# --------------------
# We use the AdaBoost boosting algorithm with a decision tree as weak learner.

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(random_state=0, n_estimators=100)
clf.fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])

#%%
# Run the check
# ==============
from deepchecks.tabular.checks import BoostingOverfit

result = BoostingOverfit().run(train_ds, validation_ds, clf)
result

#%%
# Define a condition
# ==================
# Now, we define a condition that will validate if the percent of decline between the maximal score achieved in any
# boosting iteration and the score achieved in the last iteration is above 0.02%.
check = BoostingOverfit()
check.add_condition_test_score_percent_decline_less_than(0.0002)
result = check.run(train_ds, validation_ds, clf)
result.show(show_additional_outputs=False)
