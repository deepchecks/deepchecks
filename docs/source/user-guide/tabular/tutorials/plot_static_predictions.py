# -*- coding: utf-8 -*-
"""
Using static predictions
***********************

In some cases the model evaluation can take a long time i.e very large dataset.
For this case you may calculate the predictions beforehand and pass them to the check/suite.

* If the train dataset shares indexes with the test dataset we will add train/test prefixes.
  This will cause the IndexTrainTestLeakage to not work.  
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
# ======================
# We will calculate the feature importance which is optional but will affect some displays and checks.
#
# We also calculating all the model predict and predict_proba results.

from deepchecks.utils.features import calculate_feature_importance

feature_importance, _ = calculate_feature_importance(model, test_dataset)

train_predictions = model.predict(train_dataset.features_columns)
test_predictions = model.predict(test_dataset.features_columns)
train_proba = model.predict_proba(train_dataset.features_columns)
test_proba = model.predict_proba(test_dataset.features_columns)

#%%
# Run a Deepchecks Suite Using The Static Predictions
# ===================================================
# Run the model_evaluation suite
# ------------------------------
#
# We will now pass the feature importance and the predictions we calculated before hand.

from deepchecks.tabular.suites import model_evaluation

suite = model_evaluation()

#%%

suite.run(train_dataset=train_dataset, test_dataset=test_dataset,
          features_importance=feature_importance,
          y_pred_train=train_predictions, y_pred_test=test_predictions,
          y_proba_train=train_proba, y_proba_test=test_proba)

