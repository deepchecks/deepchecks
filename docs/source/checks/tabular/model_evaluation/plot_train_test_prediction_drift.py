# -*- coding: utf-8 -*-
"""
Train Test Prediction Drift
***************************

This notebooks provides an overview for using and understanding the tabular prediction drift check.

**Structure:**

* `What Is Prediction Drift? <#what-is-prediction-drift>`__
* `Generate Data <#generate-data>`__
* `Build Model <#build-model>`__
* `Run check <#run-check>`__

What Is Prediction Drift?
========================
Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons of a machine learning model performance degrades
over time.

Prediction drift is when drift occurs in the prediction itself.
Calculating prediction drift is especially useful in cases
in which labels are not available for the test dataset, and so a drift in the predictions
is our only indication that a changed has happened in the data that actually affects model
predictions. If labels are available, it's also recommended to run the `Label Drift Check
</examples/tabular/checks/distribution/examples/plot_train_test_label_drift.html>`__.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide.rst>`

How Deepchecks Detects Prediction Drift
------------------------------------

This check detects prediction drift by using :doc:`univariate measures </user-guide/general/drift_guide.rst#detection-by-univariate-measure>`
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
encoder = LabelEncoder()
train_ds.data[label_name] = encoder.fit_transform(train_ds.data[label_name])
test_ds.data[label_name] = encoder.transform(test_ds.data[label_name])

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
