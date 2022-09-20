.. _feature_importance:

====================
Feature Importance
====================

* `What is Feature Importance? <#what-is-feature-importance>`__
* `Why Does Deepchecks Use Feature Importance? <#why-does-deepchecks-use-feature-importance>`__
* `What if the Feature Importance Calculation Takes Too Long? <#what-if-the-feature-importance-calculation-takes-too-long>`__


What is Feature Importance?
===========================

Feature importance is a ranking that represents the significance of input features to the model's predictions.
A feature with higher importance has more influence on the prediction of the model.
Feature importance can be general (meaning, for all model predictions, on average) or local (meaning, for a specific
sample).
There are many ways to calculate feature importance, some are generic for all models (such as `Shapley values <https://christophm.github.io/interpretable-ml-book/shapley.html>`_)
and some are specific for a specific model type (such as the `Gini importance <https://medium.com/the-artificial-impostor/feature-importance-measures-for-tree-models-part-i-47f187c1a2c3#:~:text=Gini%20Importance%20or%20Mean%20Decrease%20in%20Impurity%20(MDI)%20calculates%20each,number%20of%20samples%20it%20splits.>`_
for decision trees).


Why Does Deepchecks Use Feature Importance?
===========================================

Deepchecks uses your model's feature importance for 2 main reasons:

* Help you find issues with your model or data, as in the check :doc:`UnusedFeatures </checks_gallery/tabular/model_evaluation/plot_unused_features>`
* Prioritize the display according to the most relevant information the check has found (for instance, if deepchecks found drift in many features,
  as in the check :doc:`TrainTestFeatureDrift </checks_gallery/tabular/train_test_validation/plot_train_test_feature_drift>`,
  it would only display the features with the highest importance)


.. note::

	Most checks don't *require* the usage of feature importance. For those, you can
	shorten or `even skip <#what-if-the-feature-importance-calculation-takes-too-long>`__ this phase of the calculation.


How Does Deepchecks Get Feature Importance?
===========================================

There are 3 ways in which deepchecks can get your model's feature importance:


Your Model Has a Built-in Feature Importance
--------------------------------------------

First of all, deepchecks searches for your model's built-in feature importance, as some scikit-learn models have.
Deepchecks looks for the attribute ``feature_importances_`` or ``coef_`` and uses that information if it exists.


You Insert Your Own Feature Importance Data
-------------------------------------------

This can be done by using the ``features_importance`` parameter in the ``run`` function, available in all
checks and suites.
Deepchecks expects this data to be a ``pandas.Series`` where the index is feature names and the value is the calculated
importance.

>>> check = UnusedFeatures()
>>> check.run(ds_train, ds_test, model, feature_importance=pd.Series({'feat1': 0.3, 'feat2': 0.7}))

If you don't have your feature importance precalculated, you can use deepchecks to calculate it:

>>> from deepchecks.tabular.feature_importance import calculate_feature_importance
>>> fi = calculate_feature_importance(model, ds_train)
>>> check.run(ds_train, ds_test, model, feature_importance=fi)


Deepchecks Calculates the Feature Importance for You
----------------------------------------------------

If there's no built-in feature importance in the model or the user has not supplied feature importance data of their
own, deepchecks will calculate feature importance using scikit-learn's `permutation_importance <https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html>`__.

You can also force this action by using the ``feature_importance_force_permutation`` parameter in the ``run``
function, available in all checks and suites.

>>> check = TrainTestFeatureDrift()
>>> check.run(ds_train, ds_test, model, feature_importance_force_permutation=True)


What if the Feature Importance Calculation Takes Too Long?
=============================================================

Permutation feature importance is a complex calculation which can take a lot of time, depending on the number of features and
samples in your data.
However, except for certain checks, deepchecks does not require feature importance.
Therefore, if you want deepchecks to skip the calculation of feature importance, you can use the
``feature_importance_timeout`` parameter in the ``run`` function, available in all checks and suites.
Before running the permutation feature importance, deepchecks predicts the calculation time. If the predicted time
is bigger than ``feature_importance_timeout``, the process will be skipped.
Configuring this parameter to 0 will ensure the calculation is always skipped.

>>> check = MultivariateDrift()
>>> check.run(ds_train, ds_test, model, feature_importance_timeout=0)
