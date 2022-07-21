.. image:: /_static/images/general/deepchecks-logo-with-white-wide-back.png
   :target: https://deepchecks.com/?utm_source=docs.deepchecks.com&utm_medium=referral&utm_campaign=welcome
   :alt: Deepchecks Logo
   :align: center

.. image:: /_static/images/general/checks-and-conditions.png
   :alt: Deepchecks Suite of Checks
   :align: center

|

========================
Welcome to Deepchecks!
========================

Deepchecks is the leading tool for testing and for validating your machine learning models
and data, and it enables doing so with minimal effort. Deepchecks accompanies you through
various validation and testing needs such as verifying your data's integrity, inspecting its distributions,
validating data splits, evaluating your model and comparing between different models.

.. admonition:: Join Our Community 👋
   :class: tip

   In addition to perusing the documentation, please feel free to
   ask questions on our `Slack Community <https://join.slack.com/t/deepcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg>`_,
   or to post a issue or start a discussion on `Github <https://github.com/deepchecks/deepchecks/issues>`_.


💻 Installation
=================

In order to use deepchecks, you need to install it with pip:

.. code-block:: bash

    # deepchecks for tabular data:
    pip install deepchecks --upgrade

    # for installing deepchecks including the computer vision subpackage (note - Pytorch should be installed separately):
    pip install "deepchecks[vision]" --upgrade

For more installation details and best practices, check out our :doc:`full installation instructions </getting-started/installation>`.


🏃‍♀️ See It in Action
=======================

For a quick start, check out the following examples in the tutorials section, to have deepchecks up and running in a few minutes:

Tabular Data
-------------

Head over to one of our following quickstart tutorials, and have deepchecks running on your environment in less than 5 min:

- :doc:`Data Integrity Quickstart </user-guide/tabular/auto_tutorials/plot_quick_data_integrity>`

- :doc:`Train-Test Validation Quickstart </user-guide/tabular/auto_tutorials/plot_quick_train_test_validation>`

- :doc:`Model Evaluation Quickstart </user-guide/tabular/auto_tutorials/plot_quick_model_evaluation>`

 **Recommended - download the code and run it locally** on the built-in dataset and (optional) model, or **replace them with your own**.


🚀 See Our Checks Demo
^^^^^^^^^^^^^^^^^^^^^^^^^

Play with some of the existing checks in our `Interactive Checks Demo <https://checks-demo.deepchecks.com/?check=No+check+selected
&utm_source=docs.deepchecks.com&utm_medium=referral&utm_campaign=getting_started&utm_content=checks_demo_text>`__, 
and see how they work on various datasets with custom corruptions injected.


Computer Vision
----------------

.. admonition:: Note: CV Subpackage is in Beta Release

   It is :doc:`available for installation </getting-started/installation>` from PyPi, use at your own discretion.
   `Github Issues <https://github.com/deepchecks/deepchecks/issues>`_ for feedback and feature requests are welcome!

- :doc:`Object Detection Tutorial </user-guide/vision/auto_tutorials/plot_detection_tutorial>`
- :doc:`Image Data Validation in 5 Minutes </user-guide/vision/auto_tutorials/plot_simple_classification_tutorial>`
- :doc:`Classification Tutorial</user-guide/vision/auto_tutorials/plot_classification_tutorial>`



🙋🏼 When Should You Use Deepchecks?
=====================================

While you're in the research phase, and want to validate your data, find potential methodological 
problems, and/or validate your model and evaluate it.

.. image:: /_static/images/general/pipeline_when_to_validate.svg
   :alt: When To Validate - ML Pipeline Schema
   :align: center

See the :doc:`When Should You Use </getting-started/when_should_you_use>` section for an elaborate explanation of the typical scenarios.


📊 Which Types of Checks Exist?
=================================

Check out our :doc:`/checks_gallery/tabular` to see all the available checks for Tabular and
:doc:`/checks_gallery/vision` for CV.

They are checks for different phases in the ML workflow:

- Data Integrity
- Train-Test Validation (Distribution and Methodology Checks)
- Model Performance Evaluation


🧐 How Does it Work?
========================

Deepchecks is built of checks, each designated to help to identify a specific issue.
Some checks relate only to the data and labels and some require also the model.
Suites are composed of checks. Each check contains outputs to display in a notebook and/or conditions with a pass/fail/warning output.
For more information about deepchecks structure and components head over to our :doc:`/user-guide/general/deepchecks_hierarchy` in the User Guide.


❓ What Do You Need in Order to Start?
---------------------------------------

Depending on your phase and what you wish to validate, you'll need **a
subset** of the following:

-  **Raw data** (before pre-processing such as OHE, string processing,
   etc.), with optional labels
-  The model's **training data with labels**
-  **Test data** (which the model isn't exposed to) with labels
-  | A **supported model** that you wish to validate, including: **scikit-learn, XGBoost, PyTorch, and more**.
   | For tabular data models see :doc:`supported models </user-guide/tabular/supported_models>`, for more details about the supported model API.
   | For **Computer Vision**, we currently support the **PyTorch** framework. See :doc:`/user-guide/vision/data-classes/index` to understand how to integrate your data.



👀 Viewing Check and Suite Results
=====================================

The package's output can be consumed in various formats:

- Viewed inline in Jupyter (default behavior)
- :doc:`Exported as an HTML Report / JSON / Sent to W&B </user-guide/general/exporting_results/examples/index>`



🔢 Suported Data Types
=========================

Deepchecks currently supports Tabular Data (:mod:`deepchecks.tabular`) and is in beta release for Computer Vision (:mod:`deepchecks.vision`).