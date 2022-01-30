=================================
When Should You Use Deepchecks
=================================

Deepchecks has many built-in checks and suites that can be helpful for
efficiently identify potential problems in your data and models, in various points throughout the research phase.
Of course, every research process has its unique steps and challenges,
and therefore all checks and suites can easily be adapted and extended, and run separately or together.
Alongside that, we have identified that there are several recurring scenarios, that each have their own needs and characteristics.
Here we explain them shortly and note the specific pre-defined suites that are built (and enhanced on an ongoing basis)
with the purpose of efficiently giving value and initiating a quick start for each of these validation-requiring scenarios.

Typical Scenarios
==================

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

New Data: Single Dataset Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you start working with a new dataset, you have only a single dataset (no train-test split),
and you probably don't have a model.
As part of your EDA you want to ensure your data's integrity, and have it ready for your needs.
For example, you want to know if there are many duplicate samples, problems with string or categorical features,
significant outliers, inconsistent labels, etc.

For these purposes you can use the :doc:`single_dataset_integrity </api/generated/deepchecks.tabular.suites.default_suites.single_dataset_integrity.html>` suite.

.. _when_should_you_use__split_data:

After Splitting the Data: Train-Test Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When you split your data (for whichever purpose and manner), you have two or more separate datasets, however you might not have a model yet.
Just before you continue working with your data you want to ensure that the splits are indeed representative as you want them to be.
For example, you want to verify that the classes are balanced similarly, that there is no significant change in distributions between the features or labels in each of the classes,
that there is no potential data leakage that may contaminate your model or perceived results, etc.

For these purposes you can use the :doc:`train_test_validation </api/generated/deepchecks.tabular.suites.default_suites.train_test_validation.html>` suite.


.. _when_should_you_use__evaluate_model:

After Training a Model: Performance Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
At this phase you have a trained model which you want to evaluate.
Thus, you probably want to look at examine several performance metrics, compare it to various benchmarks and be able to construct a clear picture about the model's performance.
you may also want to try identify where it under-performs, and investigate to see if you discover any insights that you may use to improve its performance.

For these purposes you can use the :doc:`model_evaluation </api/generated/deepchecks.tabular.suites.default_suites.model_evaluation.html>` suite.


.. _when_should_you_use__general_overview:

General Overview: Full Suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here you want to receive all of the insights that you can get, given a specific state of the model and the data.

For this purpose you can use the :doc:`full_suite </api/generated/deepchecks.tabular.suites.default_suites.full_suite.html>`.