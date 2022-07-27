.. _supported_models:

**************************************
Supported Models and Static Prediction
**************************************

Some checks, including all the model evaluation related checks, require model predictions in order to run.
The prediction can be provided **either** within a model object or as pre-computed static predictions.

Whether provided from a model interface or as a pre-computed static predictions,
the predictions must be in the following format based on the task type:

* Classification: ``proba``, an `ndarray` of shape ``(n_samples, n_classes)`` containing the predicted
  probability of each class per sample in the dataset.
* Regression: ``pred``, an `ndarray` of shape ``(n_samples,)``, containing the predicted value
  for each sample in the dataset.

For classification tasks, it is optional to also supply ``pred`` predictions. If not supplied, Deepchecks will
assume the predicted value is the class with the highest probability.

Using a Model Interface
***********************

Supported model objects should implement at least one of the following interfaces based on the task type:

* Classification: ``predict_proba`` method which receives an |array-like|  of shape ``(n_samples, n_features)``
  and is expected to return an `ndarray` as specified above.
* Regression: ``predict`` method which receives an |array-like|  of shape ``(n_samples, n_features)``
  and is expected to return an `ndarray` as specified above.

.. Note::
    The methods are implemented using the scikit-learn api conventions therefore built-in scikit-learn
    classifiers and regressors,
    along with many additional popular models types (e.g. XGBoost, LightGBM, CatBoost etc.) are supported.

Optional Model Interface
========================

Feature Importance
-------------------

Some checks require the model's
:doc:`feature importance </api/generated/deepchecks.user-guide.tabular.feature_importance>`
for their analysis. It can be calculated
using |permutation importance| or be extracted directly from the model via property:
``feature_importances_`` or ``coef_`` for a linear model. The default behavior is to use the builtin feature
importance property if it exists, and if it doesn't, to calculate the feature importance using permutation importance.

Check-Specific Model Interfaces
--------------------------------

Some checks require specific apis to run. For example, :doc:`BoostingOverfit </api/generated/deepchecks.tabular.checks.model_evaluation.BoostingOverfit>`
requires model to be a supported boosting model type. Examples for such models include XGBoost, LightGBM, CatBoost and additional GBM implementations.

Example Regression Model
========================

>>> class simple_regression_with_importance_model:
...     def predict(X: pd.DataFrame) -> pd.Series:
...         ...
...     @property
...     def feature_importances_(self):
...         ...

Running Deepchecks with a Supported Model
=========================================

>>> from deepchecks.tabular.datasets.classification.iris import load_data, load_fitted_model
... from deepchecks.tabular.suites import model_evaluation
...
... ds_train, ds_test = load_data(data_format='Dataset')
... rf_clf = load_fitted_model() # trained sklearn RandomForestClassifier
... model_evaluation().run(train_dataset=ds_train, test_dataset=ds_test, model=rf_clf)

Using Pre-computed Predictions
******************************

There are several cases in which it's useful to provide pre-computed predictions instead of a model object.

For example, in cases where the model's prediction process takes a long time or is computationally expensive.
Another example is when the model is stored in different location or does not have a compatible API for providing
the predictions.

.. Note::
    If the train dataset shares indices with the test dataset we will add train/test prefixes to the indexes.


Running Deepchecks with Pre-computed Predictions
================================================

We will run the Deepchecks model evaluation suite using static predictions from a random forest classification model.
Since it is a classification model we will supply the ``proba``  probabilities predictions as defined above.
In addition, we will calculate and pass |permutation importance|.

>>> from deepchecks.tabular.datasets.classification.iris import load_data, load_fitted_model
... from deepchecks.tabular.suites import model_evaluation
... from deepchecks.tabular.feature_importance import calculate_feature_importance
...
... ds_train, ds_test = load_data(data_format='Dataset')
... rf_clf = load_fitted_model() # trained sklearn RandomForestClassifier
...
... fi = calculate_feature_importance(rf_clf, ds_train)
... train_proba = rf_clf.predict_proba(train_dataset.features_columns)
... test_proba = rf_clf.predict_proba(test_dataset.features_columns)
...
... model_evaluation().run(train_dataset=ds_train, test_dataset=ds_test,
        features_importance=fi, y_proba_train=train_proba, y_proba_test=test_proba)


..
    external links to open in new window

.. |array-like| raw:: html

    <a href="https://scikit-learn.org/stable/glossary.html#term-array-like" target="_blank">array-like</a>

.. |permutation importance| raw:: html

    <a href="https://scikit-learn.org/stable/modules/permutation_importance.html" target="_blank">sklearn permutation_importance</a>
