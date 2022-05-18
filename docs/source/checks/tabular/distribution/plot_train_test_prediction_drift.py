# -*- coding: utf-8 -*-
"""
Train Test Prediction Drift
***************************

This notebooks provides an overview for using and understanding the tabular prediction drift check.

**Structure:**

* `What is prediction drift? <#what-is-prediction-drift>`__
* `Generate Data <#generate-data>`__
* `Build Model <#build-model>`__
* `Run check <#run-check>`__

What Is Prediction Drift?
===========================
The term drift (and all it's derivatives) is used to describe any change in the data compared
to the data the model was trained on. Prediction drift refers to the case in which a change
in the data (data/feature drift) has happened and as a result, the distribution of the
models' prediction has changed.

Calculating prediction drift is especially useful in cases
in which labels are not available for the test dataset, and so a drift in the predictions
is our only indication that a changed has happened in the data that actually affects model
predictions. If labels are available, it's also recommended to run the `Label Drift Check
</examples/tabular/checks/distribution/examples/plot_train_test_label_drift.html>`__.

There are two main causes for prediction drift:

* A change in the sample population. In this case, the underline phenomenon we're trying
  to predict behaves the same, but we're not getting the same types of samples. For example,
  Iris Virginica stops growing and is not being predicted by the model trained to classify Iris species.
* Concept drift, which means that the underline relation between the data and
  the label has changed.
  For example, we're trying to predict income based on food spending, but ongoing inflation effect prices.
  It's important to note that concept drift won't necessarily result in prediction drift, unless it affects features that
  are of high importance to the model.

How Does the TrainTestPredictionDrift Check Work?
=================================================
There are many methods to detect drift, that usually include statistical methods
that aim to measure difference between 2 distributions.
We experimented with various approaches and found that for detecting drift between 2
one-dimensional distributions, the following 2 methods give the best results:

* For regression problems, the `ramer's V <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`__
* For classification problems, the `Wasserstein Distance (Earth Mover's Distance) <https://en.wikipedia.org/wiki/Wasserstein_metric>`__

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
