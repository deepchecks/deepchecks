.. image:: _static/deepchecks-logo-with-white-wide-back.png
   :target: https://deepchecks.com/?utm_source=docs.deepchecks.com&utm_medium=referral&utm_campaign=welcome
   :alt: Deepchecks Logo
   :align: center

.. image:: _static/checks_and_conditions.png
   :alt: Deepchecks Suite of Checks
   :align: center

|

======================
Welcome to Deepchecks!
======================

Deepchecks is the leading tool for validating your machine learning models
and data, and it enables doing so with minimal effort. Deepchecks accompanies you through
various validation needs such as verifying your data's integrity, inspecting its distributions,
validating data splits, evaluating your model and comparing between different models.

Deepchecks currently supports Tabular Data (:mod:`deepchecks.tabular`) and is in beta release for Computer Vision (:mod:`deepchecks.vision`).


See It in Action
================

For a quickstart, check out the following in the tutorials section:

**Tabular Data**:

- :doc:`Quickstart in 5 minutes </tutorials/tabular/examples/plot_quickstart_in_5_minutes>`

**Computer Vision**


.. note:: 
   Deepchecks' Computer Vision subpackage is in beta release.
   It is :doc:`available for installation </getting-started/index>` from PyPi, use at your own discretion.
   `Github Issues <https://github.com/deepchecks/deepchecks/issues>`_ are welcome!


- :doc:`Tutorial for Classification </tutorials/vision/examples/plot_classification_tutorial>`
- :doc:`Tutorial for Object Detection </tutorials/vision/examples/plot_detection_tutorial>`


Viewing Check and Suite Results
--------------------------------

The package's output can be consumed in various formats:
   - Viewed inline in Jupyter (default behavior)
   - :doc:`Exported as an HTML Report / JSON / Sent to W&B </user-guide/general/exporting_results/examples/index>`


Installation
------------
Check out our :doc:`Installation Instructions </getting-started/index>` to install it locally and continue from there.


When Should You Use Deepchecks?
================================

While you're in the research phase, and want to validate your data, find potential methodological 
problems, and/or validate your model and evaluate it.

.. image:: /_static/pipeline_when_to_validate.svg
   :alt: When To Validate - ML Pipeline Schema
   :align: center

See the :doc:`When Should You Use </getting-started/when_should_you_use>` Section for an elaborate explanation of the typical scenarios.



How Does it Work?
===================

Suites are composed of checks. Each check contains outputs to display in a notebook and/or conditions with a pass/fail/warning output.
For more information head over to our :doc:`/user-guide/general/deepchecks_hierarchy` in the User Guide.

What Do You Need in Order to Start?
=====================================

Depending on your phase and what you wish to validate, you'll need **a
subset** of the following:

-  **Raw data** (before pre-processing such as OHE, string processing,
   etc.), with optional labels
-  The model's **training data with labels**
-  **Test data** (which the model isn't exposed to) with labels
-  | A **supported model** that you wish to validate.
   | For tabular data, see :doc:`supported models </user-guide/tabular/supported_models>`.
   | For computer vision, we currently support the pytorch framework. See :doc:`/user-guide/vision/data-classes/index` to understand how to integrate your data.



See More
=========

.. note::
    In addition to perusing the documentation, please feel free to
    to ask questions on our `Slack Community <https://join.slack.com/t/deepcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg>`_,
    or to post a issue or start a discussion on `Github <https://github.com/deepchecks/deepchecks/>`_.

.. toctree::
    :maxdepth: 2

    getting-started/index

.. toctree::
    :maxdepth: 3

    tutorials/index

.. toctree::
    :maxdepth: 3

    user-guide/index


.. toctree::
    :maxdepth: 3

    examples/index


.. toctree::
    :maxdepth: 3

    api/index


For additional usage examples and for understanding the best practices of how to use the package, stay tuned,
as this package is in active development!


.. |binder badge| image:: /_static/binder-badge.svg
   :target: tutorials/tabular/quickstart_in_5_minutes.html

.. |colab badge| image:: /_static/colab-badge.svg
   :target: tutorials/tabular/quickstart_in_5_minutes.html
