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

   In addition to perusing the documentation, feel free to:

   - Ask questions on our `Slack Community <https://www.deepchecks.com/slack>`__,
   - Post an issue or start a discussion on `Github Issues <https://github.com/deepchecks/deepchecks/issues>`__.

   To support us, please give us a star ⭐️ on `Github <https://github.com/deepchecks/deepchecks>`__, it really means a lot for open source projects!


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

- :doc:`Data Integrity Quickstart </user-guide/tabular/auto_quickstarts/plot_quick_data_integrity>`

- :doc:`Train-Test Validation Quickstart </user-guide/tabular/auto_quickstarts/plot_quick_train_test_validation>`

- :doc:`Model Evaluation Quickstart </user-guide/tabular/auto_quickstarts/plot_quick_model_evaluation>`

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

- :doc:`Simple Image Classification Tutorial (for data without model) </user-guide/vision/auto_tutorials/plot_simple_classification_tutorial>`
- :doc:`Classification Tutorial</user-guide/vision/auto_tutorials/plot_classification_tutorial>`
- :doc:`Object Detection Tutorial </user-guide/vision/auto_tutorials/plot_detection_tutorial>`
- :doc:`Semantic Segmentation Tutorial</user-guide/vision/auto_tutorials/plot_segmentation_tutorial>`


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
- Train-Test Validation (Distribution, Drift and Methodology Checks)
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
   | For **Computer Vision** we support any framework, with special integrations for **PyTorch** and
   | **TensorFlow**. See :doc:`/user-guide/vision/VisionData` to understand how to integrate your data.



👀 Viewing and Saving the Results
====================================

The package's check and suite results can be consumed in various formats. Check out the following guides for more info about:

- :doc:`Viewing the results when working with Jupyter or with other IDE's </user-guide/general/showing_results>`
- :doc:`Saving an HTML report of the results </user-guide/general/export_save_results>`
- :doc:`Exporting the results (to json, or for sending the results to other tools) </user-guide/general/export_save_results>`



🔢 Supported Data Types
=========================

Deepchecks currently supports Tabular Data (:mod:`deepchecks.tabular`) and is in beta release for Computer Vision (:mod:`deepchecks.vision`).