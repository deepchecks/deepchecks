# -*- coding: utf-8 -*-
"""
Boosting Overfit
****************

This notebooks provides an overview for using and understanding feature drift check.

**Structure:**

* `What is a boosting overfit? <#what-is-a-boosting-overfit>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is A Boosting Overfit?
===========================
A boosting algorithm is a machine learning algorithm that uses a combination of weak learners to predict a target
variable. The mechanism of boosting is to increase the number of weak learners in the ensemble by adding a new
weak learner to the ensemble. The new weak learner is trained on the data that is not predicted correctly by the
ensemble in the previous iteration. The mechanism continues until the ensemble reach a certain level of accuracy or
until the number of iterations is reached.

Thanks to its mechanism, boosting algorithms are usually less prone to overfitting. However, the number of weak
learners in the ensemble can be too large making the ensemble too complex. In this case, the ensemble may be
overfitted on the training data.

How deepchecks detects a boosting overfit?
------------------------------------------
The check runs for a pre-defined number of iterations, and in each step it limits the boosting model to use up to X
estimators (number of estimators is monotonic increasing). It plots the given score calculated for each step for
both the train dataset and the test dataset.

If the percent of decline between the maximal score achieved in any boosting iteration and the score achieved in the
last iteration ("regular" model score) is above a given threshold (0.05 by default), it means the model is overfitted
and the check will fail.

Generate data & model
=====================

The dataset is the adult dataset which can be downloaded from the UCI machine
learning repository.

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from deepchecks.tabular.datasets.classification import adult
from deepchecks.tabular import Dataset

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

clf = AdaBoostClassifier(random_state=0)
clf.fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])

#%%
# Run the check
# ==============
from deepchecks.tabular.checks.methodology.boosting_overfit import BoostingOverfit

result = BoostingOverfit().run(train_ds, validation_ds, clf)
result

#%%
# Define a condition
# ==================
# Now, we define a condition that will validate if the percent of decline between the maximal score achieved in any
# boosting iteration and the score achieved in the last iteration is above 10%.
check = BoostingOverfit()
check.add_condition_test_score_percent_decline_not_greater_than(0.1)
result = check.run(train_ds, validation_ds, clf)
result.show(show_additional_outputs=False)
