.. _general__index:

=======
General
=======

Here you can find the key concepts, structure, recommended flow, and dive in to many of the deepchecks functionalities.

.. image:: /_static/images/general/checks-and-conditions.png
   :alt: Deepchecks Testing Suite of Checks
   :width: 50%
   :align: center

Concepts
--------

This section gives an overview about how to work with deepchecks: when to use it, our concepts and hierarchy.



.. toctree::
    :titlesonly:
    :maxdepth: 0
    :caption: Concepts

    concepts/when_should_you_use
    concepts/deepchecks_hierarchy

Guides
------

This section contain guides which are more general and not relevant to a specific use case or data type.
The logic explained in the guides is used throughout the package in a variety of locations.


.. toctree::
    :titlesonly:
    :maxdepth: 0
    :caption: Guides

    guides/drift_guide
    guides/metrics_guide

Usage
-----

This section contain information regarding technical aspects of deepchecks usage, such as how to connect it
to your ci/cd process, different ways to save the check and suite results,
how to customize checks, suites, metrics, etc.

.. toctree::
    :titlesonly:
    :maxdepth: 0
    :caption: Usage

    usage/ci_cd
    usage/customizations/auto_examples/index
    usage/showing_results
    usage/export_save_results


.. _user_guide__integrations:

Integrations
------------

Here you can see code examples for how to use deepchecks with various existing tools.
Of course, deepchecks can easily be integrated with many additional tools, here you can find
examples and code snippets for inspiration. Contributions to this docs section are very welcome!

.. toctree::
    :titlesonly:
    :maxdepth: 0
    :caption: Integrations

    integrations/spark_databricks
    integrations/pytest
    integrations/h2o
    integrations/hugging_face
    integrations/airflow
    integrations/cml
    integrations/junit
    usage/exporting_results/auto_examples/plot_exports_output_to_wandb
