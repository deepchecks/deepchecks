# -*- coding: utf-8 -*-
"""
Train Test Prediction Drift
***************************
This notebooks provides an overview for using and understanding the tabular prediction drift check.

**Structure:**

* `What is a prediction drift? <#what-is-a-prediction-drift>`__
* `Run check on a Classification task <#run-the-check-on-a-classification-task-mnist>`__
* `Run check on an Object Detection task <#run-the-check-on-an-object-detection-task-coco>`__

What Is a Prediction Drift?
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
  Iris Virginica stops growing and not be predicted by a model that classifies Iris.
* Concept drift, which means that the underline relation between the data and
  the label has changed.
  For example, inflation effect prices on data that predicts a income based on food spending.
  Important to note that concept drift won't necessarily result in prediction drift, unless it affects features that
  are of high importance to the model.

How Does the TrainTestPredictionDrift Check Work?
=================================================
There are many methods to detect drift, that usually include statistical methods
that aim to measure difference between 2 distributions.
We experimented with various approaches and found that for detecting drift between 2
one-dimensional distributions, the following 2 methods give the best results:

* For numerical features, the `Population Stability Index (PSI) <https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf>`__
* For categorical features, the `Wasserstein Distance (Earth Mover's Distance) <https://en.wikipedia.org/wiki/Wasserstein_metric>`__

"""

#%%


import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestPredictionDrift
import pprint


#%%
# Generate data
# =============

np.random.seed(42)

train_data = np.concatenate([np.random.randn(1000,2), np.random.choice(a=['apple', 'orange', 'banana'], p=[0.5, 0.3, 0.2], size=(1000, 2))], axis=1)
test_data = np.concatenate([np.random.randn(1000,2), np.random.choice(a=['apple', 'orange', 'banana'], p=[0.5, 0.3, 0.2], size=(1000, 2))], axis=1)

df_train = pd.DataFrame(train_data, columns=['numeric_without_drift', 'numeric_with_drift', 'categorical_without_drift', 'categorical_with_drift'])
df_test = pd.DataFrame(test_data, columns=df_train.columns)

df_train = df_train.astype({'numeric_without_drift': 'float', 'numeric_with_drift': 'float'})
df_test = df_test.astype({'numeric_without_drift': 'float', 'numeric_with_drift': 'float'})


#%%


df_test['numeric_with_drift'] = df_test['numeric_with_drift'].astype('float') + abs(np.random.randn(1000)) + np.arange(0, 1, 0.001) * 4
df_test['categorical_with_drift'] = np.random.choice(a=['apple', 'orange', 'banana', 'lemon'], p=[0.5, 0.25, 0.15, 0.1], size=(1000, 1))


#%%
# Model
# =====


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from deepchecks.tabular import Dataset


#%%


model = Pipeline([
    ('handle_cat', ColumnTransformer(
        transformers=[
            ('num', 'passthrough',
             ['numeric_with_drift', 'numeric_without_drift']),
            ('cat',
             Pipeline([
                 ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
             ]),
             ['categorical_with_drift', 'categorical_without_drift'])
        ]
    )),
    ('model', DecisionTreeClassifier(random_state=0, max_depth=2))]
)


#%%


label = np.random.randint(0, 2, size=(df_train.shape[0],))
cat_features = ['categorical_without_drift', 'categorical_with_drift']
df_train['target'] = label
train_dataset = Dataset(df_train, label='target', cat_features=cat_features)

model.fit(train_dataset.data[train_dataset.features], label)

label = np.random.randint(0, 2, size=(df_test.shape[0],))
df_test['target'] = label
test_dataset = Dataset(df_test, label='target', cat_features=cat_features)


#%%
# Run check
# =========


check = TrainTestPredictionDrift()
result = check.run(train_dataset=train_dataset, test_dataset=test_dataset, model=model)
result

