.. _user_guide__key_concepts:

==============
Key Concepts
==============

Check
========

Each check enables you to inspect a specific aspect of your data and models.
They are the basic building block of the deepchecks package, covering all kinds of common issues,
such as: PerformanceOverfit, DataSampleLeakage, SingleFeatureContribution, DataDuplicates, and :ref:`many more checks <examples/checks>`.
Each check can have two types of results:

#. A visual result meant for display (e.g. a figure or a table).
#. A return value that can be used for validating the expected check results
   (validations are typically done by adding a "condition" to the check, as explained below)


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

A suite is an ordered collection of checks, that can have conditions added to them. The Suite enables displaying a concluding report for all of the Checks that ran. `Here <https://github.com/deepchecks/deepchecks/blob/main/deepchecks/suites>`__ you can find the predefined existing suites and a code example demonstrating how to build your own custom suite. The existing suites include default conditions added for most of the checks. You can edit the preconfigured suites or build a suite of your own with a collection of checks and optional conditions.




