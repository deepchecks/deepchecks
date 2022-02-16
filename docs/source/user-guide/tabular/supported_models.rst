.. _supported_models:

====================
Supported Models
====================

Many checks require passing a model object. These include all model-evaluation related checks,
and in general any check that requires the model's predictions for its analysis.

.. Note::
    In order to be compatible with deepchecks, the model object should adhere to one requirement:

    - It has to have a a method enabling **prediction**

More specifically, the model should have a ``predict`` method for regression tasks, 
and for classification tasks also a ``predict_proba`` method, 
which should be implemented using the scikit-learn api conventions. 
Some checks may attempt using additional model methods if those exist, 
for more information see :ref:`Optional Model Interface <supported_models__optional_model_interface>`.

Note that built-in scikit-learn classifiers and regressors, 
along with many additional popular models types (e.g. XGBoost, LightGBM, CatBoost etc.) implement these methods 
and are thus supported.


Required Model Interface
==========================

Regression
----------

All that is necessary for a regression model is the ``predict`` function.
The predict function should expect an |array-like|  of shape ``(n_samples, n_features)``
and is expected to return an `ndarray` of shape ``(n_samples,)``, a vector containing the predicted value for each sample.

Example of a simple regression model:

>>> class simple_regression_model:
...     def predict(X: pd.DataFrame) -> pd.Series:
...         ...

Classification
--------------

For classification models, we require both the  the ``predict`` and the ``predict_proba`` function.
They both should expect an |array-like| of shape ``(n_samples, n_features)``, but ``predict``
is expected to return an `ndarray` of shape ``(n_samples,)``, a vector containing the predicted class label for each sample, and ``predict_proba``
is expected to return an `ndarray` of shape ``(n_samples, n_classes)``, an array containing the predicted probability of each class per sample.



>>> class simple_classification_model:
...     def predict(X: np.ndarray) -> np.ndarray:
...         ...
...     def predict_proba(X: np.ndarray) -> np.ndarray:
...         ...


.. _supported_models__optional_model_interface:

Optional Model Interface 
===========================

Feature Importance
-------------------

Deepchecks can calculate feature importance using |permutation importance|, and it also supports the builtin feature importance property: ``feature_importances_`` or ``coef_`` for a linear model.
The default behavior is to use the builtin feature importance property if it exists, and if it doesn't, we calculate the feature importance using permutation importance.


>>> class simple_importance_model:
...     def predict(X: pd.DataFrame) -> pd.Series:
...         ...
...     @property
...     def feature_importances_(self):
...         ...

Check-Specific Model Interfaces
--------------------------------

Some checks require specific apis to run. For example, :doc:`BoostingOverfit </api/generated/deepchecks.tabular.checks.methodology.boosting_overfit.BoostingOverfit>`
requires model to be a supported boosting model type. Examples for such models include XGBoost, LightGBM, CatBoost and additional GBM implementations.

..
    external links to open in new window

.. |array-like| raw:: html

    <a href="https://scikit-learn.org/stable/glossary.html#term-array-like" target="_blank">array-like</a>

.. |permutation importance| raw:: html

    <a href="https://scikit-learn.org/stable/modules/permutation_importance.html" target="_blank">sklearn permutation_importance</a>
