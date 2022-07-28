# -*- coding: utf-8 -*-
"""
==============================
Using Pre-computed Predictions
==============================

There are several cases in which it's useful to compute the model predictions before using deepchecks.

In some cases the model evaluation can take a long time, i.e. for a very large dataset.
This feature can also be helpful if the Model inference happens on a production node and you can get the predictions
using an api.
You can also use this feature to run deepchecks on models that do not support the sklearn api.

.. note::
    If the train dataset shares indices with the test dataset we will add train/test prefixes.
    This will cause the IndexTrainTestLeakage condition to pass even when leakage is present in cases where the
    DataFrame index is also defined as the deepchecks Dataset index.
"""

#%%
# Generate data & model
# =====================

from deepchecks.tabular.datasets.classification.iris import (
    load_data, load_fitted_model)

train_dataset, test_dataset = load_data()
model = load_fitted_model()

#%%
# Set Up The Required Data
# ========================
# We will calculate the feature importance which is optional but will affect some displays and checks.
# (feature importance can also be provided from other sources (e.g. using a custom model that has FI as a property, shap, etc))
#
# We are also calculating all the model predict_proba results.
#
# For regression we would provide the predict result, we can also provide the predict result explicitly for classification (on default argmax will be used to calculate them).
# In order to pass the decisions, we can provide them using the `y_pred_train` and `y_pred_test` arguments.

from deepchecks.utils.features import _calculate_feature_importance

feature_importance, _ = _calculate_feature_importance(model, test_dataset)

train_proba = model.predict_proba(train_dataset.features_columns)
test_proba = model.predict_proba(test_dataset.features_columns)

#%%
# Run a Deepchecks Suite Using The Static Predictions
# ===================================================
#
# Run the model_evaluation suite
# ------------------------------
#
# We will now pass the feature importance and the predictions we calculated beforehand.

from deepchecks.tabular.suites import model_evaluation

suite = model_evaluation()
result = suite.run(train_dataset=train_dataset, test_dataset=test_dataset,
                   features_importance=feature_importance,
                   y_proba_train=train_proba, y_proba_test=test_proba)


#%%
# Observing the results:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The results can be saved as a html file with the following code:

result.save_as_html('output.html')

result = suite.run(train_dataset=train_dataset, test_dataset=test_dataset,
                   features_importance=feature_importance,
                   y_proba_train=train_proba, y_proba_test=test_proba)

#%%
# Export the results to HTML report
# ---------------------------------

result.save_as_html('report.html')

result
