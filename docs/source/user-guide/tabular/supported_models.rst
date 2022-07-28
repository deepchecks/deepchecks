.. _supported_models:

===================================
Working with Models and Predictions
===================================

Some checks, specially the model evaluation related checks, require model predictions in order to run.
In Deepchecks, predictions are passed into the suite / check ``run`` method in one of the following ways:

* Passing a :ref:`model object <passing_a_model>` that will compute the
  predictions on the input data.
* Passing :ref:`pre-computed predictions <using_pre-computed_predictions>`.

Passing pre-computed predictions is simple low code alternative to passing a model. It is specifically recommended to use
this option if your model object is unavailable locally (for example if placed on a separate prediction server)
or if the predicting process is computationally expensive or time consuming.

Predictions Format
==================

Deepchecks currently supports model predictions for regression, binary and multiclass classification tasks.
Whether provided from a model interface or as a pre-computed predicted values,
the predictions must be in the following format based on the task type:

* Predicted values should be provided as an `ndarray` of shape ``(n_samples,)``, containing the predicted value
  for each sample in the dataset. Predicted values are required for all task types.
* Probabilities per class should be provided as an `ndarray` of shape ``(n_samples, n_classes)``
  containing the predicted probability of each class per sample. Probabilities per class are only required for
  classification tasks.

Passing a Model
===============

Deepchecks requires models to follow the |sklearn api conventions| for calculating predicted values
and probabilities per class. Therefore built-in scikit-learn classifiers and regressors,
along with many additional popular models types (e.g. XGBoost, LightGBM, CatBoost etc.) are supported out of the box.

Specifically, Deepchecks requires the following methods to be implemented in the model object:

* ``predict`` method which receives an |array-like|  of shape ``(n_samples, n_features)`` containing the
  input features and returns :ref:`predicted values <predictions_format>`.
* ``predict_proba`` method which receives an |array-like|  of shape ``(n_samples, n_features)`` containing the
  input features and returns :ref:`probabilities per class <predictions_format>`.
  This method is required only for classification tasks.


.. note::
    If your model do not support those interfaces, you can either create a wrapper class that implements the
    required methods by calling the relevant APIs from your models or by implementing them directly inside your model
    class.

Example Custom Classification Model Wrapper
-------------------------------------------

>>> class SimpleClassificationModelWrapper:
...     def predict(X: pd.DataFrame) -> np.ndarray:
...         # Implement based on base model's API
...         ...
...     def predict_proba(X: pd.DataFrame) -> np.ndarray: # only required for classification tasks.
...         # Implement based on base model's API
...         ...
...     @property
...     def feature_importances_(self):
...         ...

Optional Model Interfaces
-------------------------

Feature Importance
~~~~~~~~~~~~~~~~~~

Some checks require the model's
:doc:`feature importance </api/generated/deepchecks.user-guide.tabular.feature_importance>`
for their analysis. It can be calculated
using |permutation importance| or be extracted directly from the model via property:
``feature_importances_`` or ``coef_`` for a linear model. The default behavior is to use the builtin feature
importance property if it exists, and if it doesn't, to calculate the feature importance using permutation importance.

Check-Specific Model Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some checks require specific apis to run. For example, :doc:`BoostingOverfit </api/generated/deepchecks.tabular.checks.model_evaluation.BoostingOverfit>`
requires model to be a supported boosting model type. Examples for such models include XGBoost, LightGBM, CatBoost and additional GBM implementations.

Running Deepchecks with a Supported Model
-----------------------------------------

.. literalinclude:: ../../../../examples/examples_supported_models.py
    :language: python
    :lines: 0-7
    :tab-width: 0

Using Pre-computed Predictions
==============================

As previously discussed, passing pre-computed predictions can be a good alternative to passing a model object in
many scenarios.

The predictions should be passed via the y_proba and y_pred arguments of the suite / check run method in
the :ref:`appropriate format <predictions_format>`. y_pred receives the predicted values of the model
and y_proba receives the probabilities per class, which is only required for classification tasks.

The predictions should be provided for each dataset supplied to the suite / check. For example the
:doc:`Simple Model Comparison </api/generated/deepchecks.tabular.checks.model_evaluation.SimpleModelComparison>`
check for a **regression** model
requires both train and test predicted values to be provided via the y_pred_train, y_pred_test arguments.

For classification tasks, predicted values are not required. If not supplied,
Deepchecks will assume the predicted class is the class with the highest predicted probability.

.. Note::
    When using pre-computed predictions, if the train dataset shares indices with the test dataset we
    will add train/test prefixes to the indexes.


Running Deepchecks with Pre-computed Predictions
------------------------------------------------

We will run the Deepchecks model evaluation suite using pre-computed predictions from a random forest classification
model. In addition, we will calculate and pass |permutation importance| which provides a better estimate of the
effect of different features on the model's performance. see
:doc:`Feature Importance </api/generated/deepchecks.user-guide.tabular.feature_importance>` for more details.

.. literalinclude:: ../../../../examples/examples_supported_models.py
    :language: python
    :lines: 7-23
    :tab-width: 0


..
    external links to open in new window

.. |array-like| raw:: html

    <a href="https://scikit-learn.org/stable/glossary.html#term-array-like" target="_blank">array-like</a>

.. |permutation importance| raw:: html

    <a href="https://scikit-learn.org/stable/modules/permutation_importance.html" target="_blank">sklearn permutation_importance</a>

.. |sklearn api conventions| raw:: html
    <a href="https://scikit-learn.org/stable/developers/develop.html" target="_blank">scikit-learn api conventions</a>