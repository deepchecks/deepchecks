.. raw:: html

   <!--
     ~ ----------------------------------------------------------------------------
     ~ Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
     ~
     ~ This file is part of Deepchecks.
     ~ Deepchecks is distributed under the terms of the GNU Affero General
     ~ Public License (version 3 or later).
     ~ You should have received a copy of the GNU Affero General Public License
     ~ along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
     ~ ----------------------------------------------------------------------------
     ~
   -->


.. raw:: html

   <p align="center">
     &emsp;
     <a href="https://join.slack.com/t/deepcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg">Join&nbsp;Slack</a>
     &emsp; | &emsp; 
     <a href="https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=top_links">Documentation</a>
     &emsp; | &emsp; 
     <a href="https://deepchecks.com/blog/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=top_links">Blog</a>
     &emsp; | &emsp;  
     <a href="https://twitter.com/deepchecks">Twitter</a>
     &emsp;
   </p>
   
.. raw:: html

   <p align="center">
      <a href="https://deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=logo">
      <img src="docs/images/deepchecks-logo-with-white-wide-back.png">
      </a>
   </p>


============================================
Test Suites for Validating ML Models & Data
============================================

|build| |Documentation Status| |pkgVersion| |pyVersions|
|Maintainability| |Coverage Status|

Deepchecks is a Python package for comprehensively validating your
machine learning models and data with minimal effort. This includes
checks related to various types of issues, such as model performance,
data integrity, distribution mismatches, and more.

Installation
=============

Using pip
----------

.. code:: bash

   pip install deepchecks -U --user

Using conda
------------

.. code:: bash

   conda install -c conda-forge deepchecks


Try it Out!
============

Head over to the `Quickstart Notebook <https://docs.deepchecks.com/en/stable/
examples/guides/quickstart_in_5_minutes.html?
utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out>`__
and choose the  |binder badge image|  or the  |colab badge image|  to have it up and running, and to then apply it on your own data and models.


Usage Examples
===============

Running a Suite
----------------
A `Suite <#suite>`_ runs a collection of `Checks <#check>`_ with
optional `Conditions <#condition>`_ added to them.

Example for running a suite on given `datasets`_ and with a `supported model`_:

.. code:: python

   from deepchecks.suites import full_suite
   suite = full_suite()
   suite.run(train_dataset=train_dataset, test_dataset=test_dataset, model=model)

Which will result in a report that looks like this:

.. raw:: html

   <p align="center">
      <img src="docs/images/full_suite_output.gif" width="750">
   </p>

See the `full example here`_.

Note that some other suites (e.g. ``single_dataset_integrity``) don't require a model as part of the input.

.. _full example here:
   https://docs.deepchecks.com/en/stable/
   examples/guides/quickstart_in_5_minutes.html?
   utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out

.. _datasets:
   https://docs.deepchecks.com/en/stable/
   user-guide/dataset_object.html
   ?utm_source=github.com&utm_medium=referral&
   utm_campaign=readme&utm_content=running_a_suite

.. _supported model:
   https://docs.deepchecks.com/en/stable/
   user-guide/supported_models.html
   ?utm_source=github.com&utm_medium=referral&
   utm_campaign=readme&utm_content=running_a_suite 


In the following section you can see an example of how the output of a single check without a condition may look.

Running a Check
----------------

To run a specific single check, all you need to do is import it and then
to run it with the required (check-dependent) input parameters. More
details about the existing checks and the parameters they can receive
can be found in our `API Reference`_.

.. _API Reference:
   https://docs.deepchecks.com/en/stable/
   api/index.html?
   utm_source=github.com&utm_medium=referral&
   utm_campaign=readme&utm_content=running_a_check

.. code:: python

   from deepchecks.checks import TrainTestFeatureDrift
   import pandas as pd

   train_df = pd.read_csv('train_data.csv')
   test_df = pd.read_csv('test_data.csv')
   # Initialize and run desired check
   TrainTestFeatureDrift().run(train_data, test_data)

Will produce output of the type:

   .. raw:: html

      <h4>Train Test Drift</h4>
      <p>The Drift score is a measure for the difference between two distributions,
      in this check - the test and train distributions. <br>
      The check shows the drift score and distributions for the features,
      sorted by feature importance and showing only the top 5 features, according to feature importance.
      If available, the plot titles also show the feature importance (FI) rank.</p>
      <p align="left">
        <img src="docs/images/train-test-drift-output.png">
      </p>

Key Concepts
==============

Check
------

Each check enables you to inspect a specific aspect of your data and
models. They are the basic building block of the deepchecks package,
covering all kinds of common issues, such as:

- Model Error Analysis
- Label Ambiguity
- Data Sample Leakage
and `many more checks`_.

.. _many more checks:
   https://docs.deepchecks.com/en/stable/
   api/checks/index.html
   ?utm_source=github.com&utm_medium=referral&
   utm_campaign=readme&utm_content=key_concepts__check

Each check can have two types of
results:

1. A visual result meant for display (e.g. a figure or a table).
2. A return value that can be used for validating the expected check
   results (validations are typically done by adding a "condition" to
   the check, as explained below).

Condition
---------

A condition is a function that can be added to a Check, which returns a
pass ✓, fail ✖ or warning ! result, intended for validating the Check's
return value. An example for adding a condition would be:

.. code:: python

   from deepchecks.checks import BoostingOverfit
   BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(threshold=0.05)

which will return a check failure when running it if there is a difference of
more than 5% between the best score achieved on the test set during the boosting
iterations and the score achieved in the last iteration (the model's "original" score
on the test set).

Suite
------

An ordered collection of checks, that can have conditions added to them.
The Suite enables displaying a concluding report for all of the Checks
that ran.

See the list of `predefined existing suites`_ for tabular data
to learn more about the suites you can work with directly and also to
see a code example demonstrating how to build your own custom suite.

The existing suites include default conditions added for most of the checks.
You can edit the preconfigured suites or build a suite of your own with a collection
of checks and optional conditions.

.. _predefined existing suites: deepchecks/tabular/suites

.. include:: 

.. raw:: html

   <p align="center">
      <img src="/docs/images/diagram.svg">
   </p>


What Do You Need in Order to Start Validating?
----------------------------------------------

Environment
~~~~~~~~~~~~

- The deepchecks package installed

- JupyterLab or Jupyter Notebook

Data / Model 
~~~~~~~~~~~~

Depending on your phase and what you wish to validate, you'll need a
subset of the following:

-  Raw data (before pre-processing such as OHE, string processing,
   etc.), with optional labels

-  The model's training data with labels

-  Test data (which the model isn't exposed to) with labels

-  A `supported model`_ (e.g. scikit-learn models, XGBoost, any model implementing the `predict` method in the required format)

Supported Data Types
~~~~~~~~~~~~~~~~~~~~

Currently the package supports tabular data.
Stay tuned for the upcoming Computer Vision release.

When Should I Run Deepchecks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deepchecks validation accompanies you from the initial phase when you
have only raw data, through the data splits, and to the final stage of
having a trained model that you wish to evaluate. See more about typical usage scenarios and the built-in suites in the
`docs <https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utme_content=what_do_you_need_in_order_to_start_validating>`__.


Documentation
--------------

-  `https://docs.deepchecks.com/ <https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=documentation>`__
   - HTML documentation (stable release)
-  `https://docs.deepchecks.com/en/latest <https://docs.deepchecks.com/en/latest/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=documentation>`__
   - HTML documentation (latest release)

Community
==========

-  Join our `Slack
   Community <https://join.slack.com/t/deepcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg>`__
   to connect with the maintainers and follow users and interesting
   discussions
-  Post a `Github
   Issue <https://github.com/deepchecks/deepchecks/issues>`__ to suggest
   improvements, open an issue, or share feedback.


.. |build| image:: https://github.com/deepchecks/deepchecks/actions/workflows/build.yml/badge.svg
.. |Documentation Status| image:: https://readthedocs.org/projects/deepchecks/badge/?version=latest
   :target: https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=badge
.. |pkgVersion| image:: https://img.shields.io/pypi/v/deepchecks
.. |pyVersions| image:: https://img.shields.io/pypi/pyversions/deepchecks
.. |Maintainability| image:: https://api.codeclimate.com/v1/badges/970b11794144139975fa/maintainability
   :target: https://codeclimate.com/github/deepchecks/deepchecks/maintainability
.. |Coverage Status| image:: https://coveralls.io/repos/github/deepchecks/deepchecks/badge.svg?branch=main
   :target: https://coveralls.io/github/deepchecks/deepchecks?branch=main

.. |binder badge image| image:: /docs/source/_static/binder-badge.svg
   :target: https://docs.deepchecks.com/en/stable/examples/guides/quickstart_in_5_minutes.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out_binder
.. |colab badge image| image:: /docs/source/_static/colab-badge.svg
   :target: https://docs.deepchecks.com/en/stable/examples/guides/quickstart_in_5_minutes.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out_colab
