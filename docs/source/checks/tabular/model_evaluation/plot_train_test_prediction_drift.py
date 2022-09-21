# -*- coding: utf-8 -*-
"""
.. _plot_tabular_train_test_prediction_drift:

Train Test Prediction Drift
***************************

This notebook provides an overview for using and understanding the tabular prediction drift check.

**Structure:**

* `What Is Prediction Drift? <#what-is-prediction-drift>`__
* `Generate Data <#generate-data>`__
* `Build Model <#build-model>`__
* `Run check <#run-check>`__

What Is Prediction Drift?
=========================

Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Prediction drift is when drift occurs in the prediction itself.
Calculating prediction drift is especially useful in cases
in which labels are not available for the test dataset, and so a drift in the predictions
is our only indication that a changed has happened in the data that actually affects model
predictions. If labels are available, it's also recommended to run the
:doc:`Label Drift check </checks_gallery/tabular/train_test_validation/plot_train_test_label_drift>`.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide>`.

How Deepchecks Detects Prediction Drift
---------------------------------------

This check detects prediction drift by using :ref:`univariate measures <drift_detection_by_univariate_measure>`
on the prediction output.
"""

#%%


from sklearn.preprocessing import LabelEncoder

from deepchecks.tabular.checks import TrainTestPredictionDrift
from deepchecks.tabular.datasets.classification import adult

#%%
# Generate data
# =============

label_name = 'income'
train_ds, test_ds = adult.load_data()

#%%
# Introducing drift:

test_ds.data['education-num'] = 13
test_ds.data['education'] = ' Bachelors'


#%%
# Build Model
# ===========


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

#%%


numeric_transformer = SimpleImputer()
categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())]
)

train_ds.features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, train_ds.numerical_features),
        ("cat", categorical_transformer, train_ds.cat_features),
    ]
)

model = Pipeline(steps=[("preprocessing", preprocessor), ("model", RandomForestClassifier(max_depth=5, n_jobs=-1))])
model = model.fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])

#%%
# Run check
# =========

check = TrainTestPredictionDrift()
result = check.run(train_dataset=train_ds, test_dataset=test_ds, model=model)
result

#%%
# The prediction drift check can also calculate drift on the predicted classes rather than the probabilities. This is
# the default behavior for multiclass tasks. To force this behavior for binary tasks, set the ``drift_mode`` parameter
# to ``prediction``.

check = TrainTestPredictionDrift(drift_mode='prediction')
result = check.run(train_dataset=train_ds, test_dataset=test_ds, model=model)
result
