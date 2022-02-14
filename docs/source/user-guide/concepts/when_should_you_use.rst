=============================================
When Should You Validate & Built-In Suites
=============================================

.. currentmodule:: deepchecks.tabular.suites.default_suites

Deepchecks has many built-in checks and suites that can help validating various points throughout the research phase.
Of course, every research process has its unique steps and challenges, and therefore all checks and suites can easily customized.
Alongside that, we have identified that there are several recurring scenarios, that each have their own needs and characteristics.

.. image:: /_static/pipeline_when_to_validate.svg
   :alt: When To Validate - ML Pipeline Schema
   :align: center

Here we explain them shortly and note the specific pre-defined suites that are built
with the purpose of efficiently giving value and initiating a quick start for each of these validation-requiring scenarios.


Built-In Suites - API Reference
================================

Check the :mod:`deepchecks.tabular.suites.default_suites` in the API reference for a list of all of the built-in suites for tabular data.


Typical Validation Scenarios
===============================

#. When you **start working with a new dataset**: :ref:`Validate New Data <when_should_you_use__new_data>`.
#. When you **split the data** (before training / various cross-validation split / hold-out test set/ ...): :ref:`Validate the Split <when_should_you_use__split_data>`.
#. When you **evaluate a model**: :ref:`Validate Model Performance <when_should_you_use__evaluate_model>`.
#. When you want to **compare different models**: *Coming Soon*.
#. A more general scenario - when you want to have a **quick overview** of a project.
   For example if you want to get back to a project that you haven't worked on for a while,
   or to go over a current project with a peer or a manager, you may want to have all of
   the information organized together: :ref:`General Overview <when_should_you_use__general_overview>`.


Of course, for each of these phases your needs and the availabe assets are different. Let's have a look at them.


.. _when_should_you_use__new_data:

*New Data*: Single Dataset Validation
=========================================

When you start working with a new dataset, you have only a single dataset (no train-test split),
and you probably don't have a model.
As part of your EDA you want to ensure your data's integrity, and have it ready for your needs.
For example, you want to know if there are many duplicate samples, problems with string or categorical features,
significant outliers, inconsistent labels, etc.

For these purposes you can use the :func:`single_dataset_integrity` suite.

.. _when_should_you_use__split_data:

*After Splitting the Data*: Train-Test Validation
====================================================

When you split your data (for whichever purpose and manner), you have two or more separate datasets, however you might not have a model yet.
Just before you continue working with your data you want to ensure that the splits are indeed representative as you want them to be.
For example, you want to verify that the classes are balanced similarly, that there is no significant change in distributions between the features or labels in each of the classes,
that there is no potential data leakage that may contaminate your model or perceived results, etc.

For these purposes you can use the :func:`train_test_validation` suite.


.. _when_should_you_use__evaluate_model:

*After Training a Model*: Analysis & Validation
====================================================

At this phase you have a trained model which you want to evaluate.
Thus, you probably want to look at examine several performance metrics, compare it to various benchmarks and be able to construct a clear picture about the model's performance.
you may also want to try identify where it under-performs, and investigate to see if you discover any insights that you may use to improve its performance.

For these purposes you can use the :func:`model_evaluation` suite.


.. _when_should_you_use__general_overview:

*General Overview*: Full Suite
==================================

Here you want to have a quick overview of the project, and receive all of the insights that you can get, given a specific state of the model and the data.

For this purpose you can use the :func:`full_suite`.