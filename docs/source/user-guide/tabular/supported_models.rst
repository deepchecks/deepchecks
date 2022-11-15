.. _supported_models:

===================================
Working with Models and Predictions
===================================

Some checks, mainly the ones related to model evaluation, require model predictions in order to run.
In deepchecks, predictions are passed into the suite / check ``run`` method in one of the following ways:

* Passing a :ref:`model object <supported_models__passing_a_model>` that will compute the
  predictions on the input data.
* Passing :ref:`pre-computed predictions <supported_models_using_pre-computed_predictions>`.

Passing pre-computed predictions is a simple alternative to passing a model. It is specifically recommended to use
this option if your model object is unavailable locally (for example if placed on a separate prediction server)
or if the predicting process is computationally expensive or time consuming.

.. _supported_models__predictions_format:

Supported Tasks and Predictions Format
======================================

Deepchecks currently supports model predictions for regression, binary and multiclass classification tasks.
Whether provided from a model interface or as a pre-computed predicted values,
the predictions must be in the following format based on the task type:

* **Predicted values**: should be provided as an |array-like| of shape ``(n_samples,)``, containing the predicted value
  for each sample in the dataset. Predicted values are required for all task types.

* **Probabilities per class**: should be provided as an |array-like| of shape ``(n_samples, n_classes)``
  containing the predicted probability of each possible class for each sample in the dataset. The probabilities
  per class should be provided in a **alphanumeric order** based on the classes names.
  Probabilities per class are only relevant for classification tasks. If predicted probabilities are not supplied,
  checks and metrics that rely on the predicted probabilities (such as ROC Curve and the AUC metric) will not run.

.. Note::
    For classification tasks, Deepchecks require the list of all possible classes in the order they appear at the
    ``probabilities per class`` vector (alphanumeric order). It can either be inferred based on provided data and model
    or supplied via the Dataset's ``label_class`` argument. For binary classification, the class with the greater
    alphanumeric value is considered the positive class.

.. _supported_models__passing_a_model:

Passing a Model
===============

Deepchecks requires models to follow the |scikit-learn API conventions| for calculating predicted values
and probabilities per class. Therefore built-in scikit-learn classifiers and regressors,
along with many additional popular models types (e.g. XGBoost, LightGBM, CatBoost etc.) are supported out of the box.

Specifically, deepchecks requires the following methods to be implemented in the model object:

* ``predict`` method which receives an |array-like|  of shape ``(n_samples, n_features)`` containing the
  input features and returns :ref:`predicted values <supported_models__predictions_format>`.
* ``predict_proba`` method which receives an |array-like|  of shape ``(n_samples, n_features)`` containing the
  input features and returns :ref:`probabilities per class <supported_models__predictions_format>`.
  This method is optional and relevant only for classification tasks.

Running Deepchecks With a Supported Model
-----------------------------------------

.. literalinclude:: ../../../../examples/examples_supported_models.py
    :language: python
    :lines: 1-6
    :tab-width: 0


Adapting Your Custom Model
--------------------------

If you are using a model that does not support those interfaces you can either add the required methods to the
model's class or create a wrapper class that implements the required interfaces by calling the relevant APIs of your
model. Below is a general structure of such wrapper class.

>>> class MyModelWrapper:
...     def predict(self, data: pd.DataFrame) -> np.ndarray:
...         # Implement based on base model's API.
...         ...
...     def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
...         # Implement based on base model's API, only required for classification tasks.
...         ...
...     @property
...     def feature_importances_(self) -> pd.Series:  # optional
...         # Return a pandas Series with feature names as index and their corresponding importance as values.
...         ...

Feature Importance (Optional)
-----------------------------

Some checks uses the model's
:doc:`feature importance </api/generated/deepchecks.user-guide.tabular.feature_importance>`
in their analysis. By default, if available, it is extracted directly from the model via property
(``feature_importances_`` or ``coef_`` for a linear model) otherwise it is calculated
using |permutation importance|. The required format for the feature importance is a pandas series with feature names
as index and their corresponding importance as values.

.. _supported_models_using_pre-computed_predictions:

Using Pre-computed Predictions
==============================

The pre-computed predictions should be passed to suite/check's ``run`` method in the appropriate format.
The parameters to pass are ``y_pred`` and ``y_proba`` for single dataset checks or ``y_pred_train`` and
``y_proba_train`` and ``y_pred_test`` and ``y_proba_test`` for checks that use both datasets.
``y_pred`` receives the predicted values of the model and ``y_proba`` receives the probabilities per class, which is
only relevant for classification tasks.
See more about the :ref:`supported formats here<supported_models__predictions_format>`.

The predictions should be provided for each dataset supplied to the suite / check. For example the
:doc:`Simple Model Comparison </api/generated/deepchecks.tabular.checks.model_evaluation.SimpleModelComparison>`
check for a regression model
requires both train and test :ref:`predicted values <supported_models__predictions_format>`
to be provided via the ``y_pred_train``, ``y_pred_test`` arguments.

For classification it's recommended but not mandatory to also pass the predicted probabilities (``y_proba``). If
predicted probabilities are not supplied, checks and metrics that rely on the predicted probabilities (such as
ROC Curve and the AUC metric) will not run.

.. Note::
    When using pre-computed predictions, if the train dataset shares indices with the test dataset we
    will add train/test prefixes to the indexes.

Code Example
------------

We will run the deepchecks model evaluation suite using pre-computed predictions from a random forest classification
model. In addition, we will calculate and pass |permutation importance| which provides a better estimate of the
effect of different features on the model's performance. See the
:doc:`feature importance API reference </api/generated/deepchecks.user-guide.tabular.feature_importance>`
for more details.

.. literalinclude:: ../../../../examples/examples_supported_models.py
    :language: python
    :lines: 8-22
    :tab-width: 0


..
    external links to open in new window

.. |array-like| raw:: html

    <a href="https://scikit-learn.org/stable/glossary.html#term-array-like" target="_blank">array-like</a>


.. |permutation importance| raw:: html

    <a href="https://scikit-learn.org/stable/modules/permutation_importance.html" target="_blank">sklearn permutation_importance</a>


.. |scikit-learn API conventions| raw:: html

    <a href="https://scikit-learn.org/stable/developers/develop.html" target="_blank">scikit-learn API conventions</a>
