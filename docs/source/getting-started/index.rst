================
Getting Started
================


💻 Installation
=================

Check out our :doc:`Installation </getting-started/installation>` instructions to install it locally and continue from there.


🏃‍♀️ See It in Action
====================

For a quick start, check out the following examples in the tutorials section:

Tabular Data
-------------

- :doc:`Quickstart in 5 minutes </tutorials/tabular/examples/plot_quickstart_in_5_minutes>`


Computer Vision
----------------

**Beta Release**

- :doc:`Deepchecks Example - Simple Image Classification Tutorial </tutorials/vision/examples/plot_simple_classification_tutorial>`
- :doc:`Deepchecks for Object Detection Tutorial </tutorials/vision/examples/plot_detection_tutorial>`
- :doc:`Deepchecks for Classification Tutorial</tutorials/vision/examples/plot_classification_tutorial>`


.. note:: 
   Deepchecks' Computer Vision subpackage is in beta release.
   It is :doc:`available for installation </getting-started/installation>` from PyPi, use at your own discretion.
   `Github Issues <https://github.com/deepchecks/deepchecks/issues>`_ are welcome!


🧐 How Does it Work?
========================

Deepchecks is built of checks, each designated to help to identify a specific issue.
Some checks relate only to the data and labels and some require also the model.
Suites are composed of checks. Each check contains outputs to display in a notebook and/or conditions with a pass/fail/warning output.
For more information about deepchecks structure and components head over to our :doc:`/user-guide/general/deepchecks_hierarchy` in the User Guide.


📊 Which Types of Checks Exist?
=================================

Check out our :doc:`/examples/index` to see all the available checks for Tabular and for CV.

They are divided in the following categories:

- Data Integrity
- Data Distribution
- Methodology
- Model Evaluation


❓ What Do You Need in Order to Start?
=======================================

Depending on your phase and what you wish to validate, you'll need **a
subset** of the following:

-  **Raw data** (before pre-processing such as OHE, string processing,
   etc.), with optional labels
-  The model's **training data with labels**
-  **Test data** (which the model isn't exposed to) with labels
-  | A **supported model** that you wish to validate.
   | For tabular data, see :doc:`supported models </user-guide/tabular/supported_models>`.
   | For computer vision, we currently support the pytorch framework. See :doc:`/user-guide/vision/data-classes/index` to understand how to integrate your data.


🙋🏼 When Should You Use Deepchecks?
=====================================

While you're in the research phase, and want to validate your data, find potential methodological 
problems, and/or validate your model and evaluate it.

.. image:: /_static/pipeline_when_to_validate.svg
   :alt: When To Validate - ML Pipeline Schema
   :align: center

See the :doc:`When Should You Use </getting-started/when_should_you_use>` Section for an elaborate explanation of the typical scenarios.


👀 Viewing Check and Suite Results
=====================================

The package's output can be consumed in various formats:

- Viewed inline in Jupyter (default behavior)
- :doc:`Exported as an HTML Report / JSON / Sent to W&B </user-guide/general/exporting_results/examples/index>`



🔢 Suported Data Types
=========================

Deepchecks currently supports Tabular Data (:mod:`deepchecks.tabular`) and is in beta release for Computer Vision (:mod:`deepchecks.vision`).



.. toctree::
    :hidden:
    :titlesonly:
    :maxdepth: 1

    index
    installation
    when_should_you_use