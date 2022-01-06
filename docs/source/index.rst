=======================
Welcome to Deepchecks!
=======================

Deepchecks is the leading tool for validating your machine learning models
and data, and it enables doing so with minimal effort. Deepchecks accompanies you through
various validation needs such as verifying your data's integrity, inspecting its distributions,
validating data splits, evaluating your model and comparing between different models.


.. image:: /_static/deepchecks-logo-with-white-wide-back.png
   :target: https://deepchecks.com/?utm_source=docs.deepchecks.com&utm_medium=referral&utm_campaign=welcome
   :alt: Deepchecks Logo
   :align: center



Get Started
============

Head over to our :doc:`/examples/guides/quickstart_in_5_minutes` tutorial,
and click on  |binder badge|  or on  |colab badge|  to launch it and see it in action,
or see our :doc:`/getting-started/index` to install it locally and continue from there.

.. note:: The package is suited for running in a jupyter environment.
          HTML and pdf reports for graphs may be added in the near future.

When Should You Use Deepchecks?
================================

While you're in the research phase, and want to validate your data, find potential methodological 
problems, and/or validate your model and evaluate it.
See the :doc:`Section in the User Guide </user-guide/when_should_you_use>` for an elaborate explanation of the typical scenarios.


Example - Validating a Model that Classifies Malicious URLs
----------------------------------------------------------------
The :doc:`following use case </examples/use-cases/phishing_urls>` demonstrates how deepchecks can be used throughout the 
research phase for model and daketa validation, enabling you to efficiently catch issues at different phases.


How Does it Work?
===================

Suites are composed of checks. Each check contains outputs to display in a notebook and/or conditions with a pass/fail/warning output.
For more information head over to our :doc:`/user-guide/key_concepts` in the User Guide.

What Do You Need in Order to Start?
=====================================

Depending on your phase and what you wish to validate, you'll need a
subset of the following:

-  Raw data (before pre-processing such as OHE, string processing,
   etc.), with optional labels
-  The model's training data with labels
-  Test data (which the model isn't exposed to) with labels
-  A model compatible with scikit-learn API that you wish to validate
   (e.g. RandomForest, XGBoost)


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
    :maxdepth: 2

    user-guide/index


.. toctree::
    :maxdepth: 2

    examples/index


.. toctree::
    :maxdepth: 3

    api/index


For additional usage examples and for understanding the best practices of how to use the package, stay tuned,
as this package is in active development!

.. |binder badge| image:: /_static/binder-badge.svg
   :target: /examples/guides/quickstart_in_5_minutes.html

.. |colab badge| image:: /_static/colab-badge.svg
   :target: /examples/guides/quickstart_in_5_minutes.html