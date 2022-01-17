.. _supported_models:

====================
Supported Models
====================

Deepchecks supports a minmal version of the sklearn api.


Required API
=============

Regression
----------

All that is necessary for for regression models is the ``predict`` function.

Example of an simple model:

>>> class simple_regression_model:
...     def predict(X: pd.DataFrame) -> pd.Series:
...         ...

Classification
--------------

For classification models, we require also the ``predict_proba`` function:

>>> class simple_classification_model:
...     def predict(X: pd.DataFrame) -> pd.Series:
...         ...
...     def predict_proba(X: pd.DataFrame) -> pd.Series:
...         ...


Other Helpful APIs
==================

Feature importance
-------------------

Deepchecks can calculate feature importance using ``permutational_importance``.
But it can support builtin feature importance property: ``feature_importances_`` or ``coef_`` for linear model:

>>> class simple_importance_model:
...     def predict(X: pd.DataFrame) -> pd.Series:
...         ...
...     @property
...     def feature_importances_(self):
...         ...

Check Specific APIs
-------------------

Some checks require specific apis to run. For example, ``BoostingOverfit`` requires model to be a `Boosting model`.

