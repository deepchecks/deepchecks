.. _supported_models:

====================
Supported Models
====================

Similar to scikit-learn API, a model will be compatible with deepchecks' checks if it implements the minimal functionality of ``predict`` and if it's classification ``predict_proba``.
Without a compatible model, deepchecks can't run checks that require predictions, for example all :doc:`performance checks </api/generated/deepchecks.checks.performance>`.

Below are the minimal requirements for each task type:

Required API
=============

Regression
----------

All that is necessary for a regression model is the ``predict`` function.
The predict function should expect an |array-like|  of shape ``(n_samples, n_features)``
and is expected to return an `ndarray` of shape ``(n_samples,)``, a vector containing the predicted value for each sample.

Example of a simple model:

>>> class simple_regression_model:
...     def predict(X: pd.DataFrame) -> pd.Series:
...         ...

Classification
--------------

For classification models, we require both the  the ``predict`` and the ``predict_proba`` functionד.
They both should expect an |array-like| of shape ``(n_samples, n_features)``, but ``predict``
is expected to return an `ndarray` of shape ``(n_samples,)``, a vector containing the predicted class label for each sample, and ``predict_proba``
is expected to return an `ndarray` of shape ``(n_samples,n_classes)``, an array containing the predicted probability of each class per sample.



>>> class simple_classification_model:
...     def predict(X: np.ndarray) -> np.ndarray:
...         ...
...     def predict_proba(X: np.ndarray) -> np.ndarray:
...         ...


Other Helpful APIs
==================

Feature importance
-------------------

Deepchecks can calculate feature importance using |permutation importance|.
But it can support builtin feature importance property: ``feature_importances_`` or ``coef_`` for linear model.


>>> class simple_importance_model:
...     def predict(X: pd.DataFrame) -> pd.Series:
...         ...
...     @property
...     def feature_importances_(self):
...         ...

Check Specific APIs
-------------------

Some checks require specific apis to run. For example, :doc:`BoostingOverfit </api/base/generated/deepchecks.checks.methodology.boosting_overfit.BoostingOverfit>`
requires model to be a `Boosting model`. Examples to such model include XGBoost, LightGBM, CatBoost or any other gbm implementation.

..
    external links to open in new window

.. |array-like| raw:: html

    <a href="https://scikit-learn.org/stable/glossary.html#term-array-like" target="_blank">array-like</a>

.. |permutation importance| raw:: html

    <a href="https://scikit-learn.org/stable/modules/permutation_importance.html" target="_blank">sklearn permutational_importance</a>
