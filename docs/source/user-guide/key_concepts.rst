==============
Key Concepts
==============

Check
========

Each check enables you to inspect a specific aspect of your data and
models. They are the basic building block of the deepchecks package,
covering all kinds of common issues, such as:

- Model Error Analysis
- Label Ambiguity
- Data Sample Leakage
and :ref:`many more checks <examples.check_demonstrations>`.


Each check can have two types of results:

1. A visual result meant for display (e.g. a figure or a table).
2. A return value that can be used for validating the expected check
   results (validations are typically done by adding a "condition" to
   the check, as explained below).


Condition
==========

A condition is a function that can be added to a Check, which returns
a pass ✓, fail ✖ or warning ! result, intended for validating the Check's return value.
An example for adding a condition would be:

.. code-block:: python

    from deepchecks.checks import BoostingOverfit
    BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(threshold=0.05)
    

Suite
========

An ordered collection of checks, that can have conditions added to them.
The Suite enables displaying a concluding report for all of the Checks
that ran. See the list of `predefined existing suites`_
to learn more about the suites you can work with directly and also to
see a code example demonstrating how to build your own custom suite.
The existing suites include default conditions added for most of the checks.
You can edit the preconfigured suites or build a suite of your own with a collection
of checks and optional conditions.

.. _predefined existing suites: deepchecks/suites




