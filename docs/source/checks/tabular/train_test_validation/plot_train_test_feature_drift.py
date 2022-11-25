# -*- coding: utf-8 -*-
"""
.. _plot_tabular_train_test_feature_drift:

Train Test Feature Drift
************************

This notebooks provides an overview for using and understanding feature drift check.

**Structure:**

* `What is a feature drift? <#what-is-a-feature-drift>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__
* `Get an aggregated value <#get-an-aggregated-value>`__

What is a feature drift?
========================
Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Feature drift is a data drift that occurs in a single feature in the dataset.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide>`.

How Deepchecks Detects Feature Drift
------------------------------------

This check detects feature drift by using :ref:`univariate measures <drift_detection_by_univariate_measure>`
on each feature column separately.
Another possible method for drift detection is by :ref:`a domain classifier <drift_detection_by_domain_classifier>`
which is used in the :doc:`Multivariate Drift check </checks_gallery/tabular/train_test_validation/plot_multivariate_drift>`.

"""

#%%
# Generate data & model
# =====================
# Let's generate a mock dataset of 2 categorical and 2 numerical features

import numpy as np
import pandas as pd

np.random.seed(42)

train_data = np.concatenate([np.random.randn(1000,2), np.random.choice(a=['apple', 'orange', 'banana'], p=[0.5, 0.3, 0.2], size=(1000, 2))], axis=1)
test_data = np.concatenate([np.random.randn(1000,2), np.random.choice(a=['apple', 'orange', 'banana'], p=[0.5, 0.3, 0.2], size=(1000, 2))], axis=1)

df_train = pd.DataFrame(train_data, columns=['numeric_without_drift', 'numeric_with_drift', 'categorical_without_drift', 'categorical_with_drift'])
df_test = pd.DataFrame(test_data, columns=df_train.columns)

df_train = df_train.astype({'numeric_without_drift': 'float', 'numeric_with_drift': 'float'})
df_test = df_test.astype({'numeric_without_drift': 'float', 'numeric_with_drift': 'float'})

#%%

df_train.head()

#%%
# Insert drift into test:
# -----------------------
# Now, we insert a synthetic drift into 2 columns in the dataset

df_test['numeric_with_drift'] = df_test['numeric_with_drift'].astype('float') + abs(np.random.randn(1000)) + np.arange(0, 1, 0.001) * 4
df_test['categorical_with_drift'] = np.random.choice(a=['apple', 'orange', 'banana', 'lemon'], p=[0.5, 0.25, 0.15, 0.1], size=(1000, 1))

#%%
# Training a model
# ----------------
# Now, we are building a dummy model (the label is just a random numerical
# column). We preprocess our synthetic dataset so categorical features are
# being encoded with an OrdinalEncoder

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
# Run the check
# =============
# Let's run deepchecks' feature drift check and see the results

from deepchecks.tabular.checks import TrainTestFeatureDrift

check = TrainTestFeatureDrift()
result = check.run(train_dataset=train_dataset, test_dataset=test_dataset, model=model)
result.show()

#%%
# Observe the check's output
# --------------------------
# As we see from the results, the check detects and returns the drift score
# per feature. As we expect, the features that were manually manipulated
# to contain a strong drift in them were detected.
#
# In addition to the graphs, each check returns a value that can be controlled
# in order to define expectations on that value (for example, to define that
# the drift score for every feature must be below 0.05).
#
# Let's see the result value for our check

result.value

#%%
# Define a condition
# ==================
# As we can see, we get the drift score for each feature in the dataset, along
# with the feature importance in respect to the model.
#
# Now, we define a condition that enforce each feature's drift score must be
# below 0.1. A condition is deepchecks' way to enforce that results are OK,
# and we don't have a problem in our data or model!

check_cond = check.add_condition_drift_score_less_than(max_allowed_categorical_score=0.2,
                                                       max_allowed_numeric_score=0.1)

#%%

result = check_cond.run(train_dataset=train_dataset, test_dataset=test_dataset)
result.show(show_additional_outputs=False)

#%%
# As we see, our condition successfully detects and filters the problematic
# features that contains a drift!
#
# Get an aggregated value
# =======================
#
# Using the :func:`reduce_output <deepchecks.tabular.checks.train_test_validation.TrainTestFeatureDrift.reduce_output>`
# function we can combine the drift values per feature and get a collective score
# that reflects the effect of the drift on the model, taking into account all the features.
# In scenarios where labels are unavailable (either temporarily of permanently)
# this value can be a good indicator of possible deterioration in the model's performance.
#
# We can define the type of aggregation we want to use via the `aggregation_method` parameter. The possible values are:
#
# ``l2_weighted`` (Default): L2 norm over the combination of drift scores and feature importance, minus the L2 norm of
# feature importance alone, specifically, ||FI + DRIFT|| - ||FI||. This method returns a value between 0 and
# sqrt(n_features). This method is built to give greater weight to features with high importance and high drift, while
# not zeroing out features with low importance and high drift.
#
# ``weighted``: Weighted mean of drift scores based on each feature's feature importance. This method
# underlying logic is that drift in a feature with a higher feature importance will have a greater effect on the model's
# performance.
#
# ``mean``: Simple mean of all the features drift scores.
#
# ``none``: No averaging. Return a dict with a drift score for each feature.
#
# ``max``: Maximum value of all the individual feature's drift scores.
#

check = TrainTestFeatureDrift(aggregation_method='weighted')
result = check.run(train_dataset=train_dataset, test_dataset=test_dataset, model=model)
result.reduce_output()
#%%
