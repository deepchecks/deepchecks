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
      <img src="docs/source/_static/images/general/deepchecks-logo-with-white-wide-back.png">
      </a>
   </p>

|build| |Documentation Status| |pkgVersion| |pyVersions|
|Maintainability| |Coverage Status|

.. raw:: html

   <h1 align="center">
      Testing and Validating ML Models & Data
   </h1>

.. raw:: html

   <p align="center">
      <a href="https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=checks_and_conditions_img">
      <img src="docs/source/_static/images/general/checks-and-conditions.png">
   </p>


🧐 What is Deepchecks?
==========================

Deepchecks is a Python package for comprehensively validating your
machine learning models and data with minimal effort. This includes
checks related to various types of issues, such as model performance,
data integrity, distribution mismatches, and more.


🖼️ Computer Vision & 🔢 Tabular Support
==========================================
**This README refers to the Tabular version** of deepchecks.

Check out the `Deepchecks for Computer Vision & Images subpackage <deepchecks/vision>`__ for more details about deepchecks for CV, currently in *beta release*.


💻 Installation
=================


Using pip
----------

.. code:: bash

   pip install deepchecks -U --user

..

   Note: Computer Vision Install

   To install deepchecks together with the **Computer Vision Submodule** that is currently in *beta release*, replace ``deepchecks`` with ``"deepchecks[vision]"`` as follows.
   
   .. code:: bash
   
      pip install "deepchecks[vision]" -U --user
   
   

Using conda
------------

.. code:: bash

   conda install -c conda-forge deepchecks


⏩ Try it Out!
================

🏃‍♀️ See It in Action
-------------------- 

Head over to one of our following quickstart tutorials, and have deepchecks running on your environment in less than 5 min:

- `Train-Test Validation Quickstart (loans data) <https://docs.deepchecks.com/stable/user-guide/tabular/
  auto_tutorials/plot_quick_data_integrity.html?
  utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out>`__

- `Data Integrity Quickstart (avocado sales data) <https://docs.deepchecks.com/stable/user-guide/tabular/
  auto_tutorials/plot_quick_data_integrity.html?
  utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out>`__

- `Full Suite (many checks) Quickstart (iris data) <https://docs.deepchecks.com/en/stable/user-guide/tabular/
  auto_tutorials/plot_quickstart_in_5_minutes.html?
  utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out>`__

 **Recommended - download the code and run it locally** on the built-in dataset and (optional) model, or **replace them with your own**.


🚀 See Our Checks Demo
------------------------

Play with some of the existing checks in our `Interactive Checks Demo <https://checks-demo.deepchecks.com/?check=No+check+selected
&utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out>`__, 
and see how they work on various datasets with custom corruptions injected.


📊 Usage Examples
====================

Running a Suite
----------------
A `Suite <#suite>`_ runs a collection of `Checks <#check>`_ with
optional `Conditions <#condition>`_ added to them.

Example for running a suite on given `datasets`_ and with a `supported model`_:

.. code:: python

   from deepchecks.tabular.suites import model_evaluation
   suite = model_evaluation()
   result = suite.run(train_dataset=train_dataset, test_dataset=test_dataset, model=model)
   result.show()

Which will result in a report that looks like this:

.. raw:: html

   <p align="center">
      <img src="docs/source/_static/images/general/full_suite_output.gif" width="750">
   </p>


Note:

- Results can also be saved as an html report, saved as json, or exported to other tools (e.g Weights & Biases - wandb)
- Other suites that run only on the data (``data_integrity``, ``train_test_validation``) don't require a model as part of the input.

See the `full code tutorials here`_.

.. _full code tutorials here:
   https://docs.deepchecks.com/dev/user-guide/tabular/auto_tutorials/index.html?
   utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out

.. _datasets:
   https://docs.deepchecks.com/en/stable/
   user-guide/tabular/dataset_object.html
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

   from deepchecks.tabular.checks import TrainTestFeatureDrift
   import pandas as pd

   train_df = pd.read_csv('train_data.csv')
   test_df = pd.read_csv('test_data.csv')
   # Initialize and run desired check
   TrainTestFeatureDrift().run(train_df, test_df)

Will produce output of the type:

   .. raw:: html

      <h4>Train Test Drift</h4>
      <p>The Drift score is a measure for the difference between two distributions,
      in this check - the test and train distributions. <br>
      The check shows the drift score and distributions for the features,
      sorted by feature importance and showing only the top 5 features, according to feature importance.
      If available, the plot titles also show the feature importance (FI) rank.</p>
      <p align="left">
        <img src="docs/source/_static/images/general/train-test-drift-output.png">
      </p>


🙋🏼  When Should You Use Deepchecks?
====================================

While you’re in the research phase, and want to validate your data, find potential methodological problems, 
and/or validate your model and evaluate it.

.. raw:: html

   <p align="center">
      <img src="/docs/source/_static/images/general/pipeline_when_to_validate.svg">
   </p>


See more about typical usage scenarios and the built-in suites in the
`docs <https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utme_content=what_do_you_need_in_order_to_start_validating>`__.



🗝️ Key Concepts
==================

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

   from deepchecks.tabular.checks import BoostingOverfit
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

.. raw:: html

   <p align="center">
      <img src="/docs/source/_static/images/general/diagram.svg">
   </p>


🤔 What Do You Need in Order to Start Validating?
==================================================

Environment
-----------

- The deepchecks package installed

- JupyterLab or Jupyter Notebook or any Python IDE


Data / Model 
------------


Depending on your phase and what you wish to validate, you'll need a
subset of the following:

-  Raw data (before pre-processing such as OHE, string processing,
   etc.), with optional labels

-  The model's training data with labels

-  Test data (which the model isn't exposed to) with labels

-  A `supported model`_ (e.g. scikit-learn models, XGBoost, any model implementing the `predict` method in the required format)


Supported Data Types
--------------------

The package currently supports tabular data and is in *beta release* for the `Computer Vision subpackage <deepchecks/vision>`__.


📖 Documentation
====================

-  `https://docs.deepchecks.com/ <https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=documentation>`__
   - HTML documentation (stable release)
-  `https://docs.deepchecks.com/en/latest <https://docs.deepchecks.com/en/latest/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=documentation>`__
   - HTML documentation (latest release)

👭 Community
================

-  Join our `Slack
   Community <https://join.slack.com/t/deepcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg>`__
   to connect with the maintainers and follow users and interesting
   discussions
-  Post a `Github
   Issue <https://github.com/deepchecks/deepchecks/issues>`__ to suggest
   improvements, open an issue, or share feedback.


.. |build| image:: https://github.com/deepchecks/deepchecks/actions/workflows/build.yml/badge.svg
.. |Documentation Status| image:: https://readthedocs.org/projects/deepchecks/badge/?version=stable
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
