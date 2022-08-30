====================
Deepchecks Hierarchy
====================

A Suite runs a collection of Checks with optional Conditions added to them.

Check
^^^^^

Each check enables you to inspect a specific aspect of your data and
models. They are the basic building block of the deepchecks package,
covering all kinds of common issues, such as:

- :doc:`/checks_gallery/tabular/model_evaluation/plot_weak_segments_performance`
- :doc:`/checks_gallery/tabular/train_test_validation/plot_train_test_feature_drift`
- :doc:`/checks_gallery/tabular/train_test_validation/plot_date_train_test_leakage_overlap`
- :doc:`/checks_gallery/tabular/data_integrity/plot_conflicting_labels`

Each check can have two types of results:

1. A visual result meant for display (e.g. a figure or a table).
2. A return value that can be used for validating the expected check
   results (validations are typically done by adding a "Condition" to
   the check, as explained below).


See Also
---------

- :doc:`Checks Demonstrations </checks_gallery/tabular/index>` - example notebooks demonstrating all of the checks implemented in the package.
- :doc:`Checks API Reference </api/deepchecks.tabular.checks>` - where all the check parameters and implemented conditions can be found.
- :doc:`/user-guide/tabular/auto_tutorials/plot_add_a_custom_check` - for a tutorial of how to implement your own check


Condition
^^^^^^^^^

A condition is a function that can be added to a Check, which returns
a pass ✓, fail ✖ or warning ! result, intended for validating the Check's return value.
An example for adding a condition would be:

.. code-block:: python

    from deepchecks.checks import BoostingOverfit
    BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(threshold=0.05)

Most checks come with pre-implemented conditions that can be added to the check, and their parameters can be tuned.
Different conditions can be concatenated on the same check. Evaluation of these conditions will happen separately for each condition.
Each condition has a ``ConditionCategory`` type, which indicates whether an unmet condition will be classified or as ``FAIL`` (✖) or as ``WARN`` (!).


See Also
---------

- :doc:`/user-guide/general/customizations/examples/plot_configure_check_conditions` - for a tutorial demonstrating how to modify a checks' conditions
- :doc:`Check Base Classes API Reference </api/deepchecks.core>` - where all of the check and condition related base objects can be found.


Suite
^^^^^

An ordered collection of checks, that can have conditions added to them.
The Suite enables displaying a concluding report for all of the Checks
that ran.
The existing suites include default conditions added for most of the checks.
You can edit the preconfigured suites or build a suite of your own with a collection
of checks and optional conditions.


See Also
---------

- `List of Tabular Suites`_ - to learn more about the tabular suites you can work with directly
- `List of Vision Suites`_ - to learn more about the vision suites you can work with directly
- :doc:`/user-guide/general/customizations/examples/plot_create_a_custom_suite` - for a demonstration about creating your own suite


.. _List of Tabular Suites: https://github.com/deepchecks/deepchecks/tree/main/deepchecks/tabular/suites
.. _List of Vision Suites: https://github.com/deepchecks/deepchecks/tree/main/deepchecks/vision/suites

.. image:: /_static/images/general/diagram.svg
   :alt: Deepchecks Diagram
   :align: center