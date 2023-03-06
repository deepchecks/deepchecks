.. _user_guide:

==========
User Guide
==========

Here you can find the key concepts, structure, recommended flow, and dive in to many of the deepchecks functionalities.


General
-------

This section section gives an overview about how to work with deepchecks: when to use it, our concepts and hierarchy, different ways
to save the check and suite results, how to customize checks, suites, metrics, etc.

.. toctree::
    :maxdepth: 2
    :caption: General

    general/when_should_you_use
    general/deepchecks_hierarchy
    general/showing_results
    general/export_save_results
    general/customizations/examples/index
    general/metrics_guide
    general/drift_guide
    general/ci_cd

Tabular
-------

Here you can see quickstarts of how to start working with deepchecks on tabular data, and much additional information 
related to the tabular supported use cases and customizations.

.. toctree::
    :maxdepth: 2
    :caption: Tabular

    tabular/auto_quickstarts/index
    tabular/auto_tutorials/index
    tabular/dataset_object
    tabular/supported_models
    tabular/feature_importance
    tabular/custom_check_templates

Vision
-------

Here you can see of how to start working with deepchecks on vision data, and much additional information 
related to the vision supported use cases and customizations.

.. toctree::
    :maxdepth: 2
    :caption: Vision

    vision/auto_tutorials/index
    vision/VisionData
    vision/supported_tasks_and_formats
    vision/vision_properties
    vision/custom_check_templates


.. _user_guide__integrations:

Integrations
------------

Here you can see code examples for how to use deepchecks with various existing tools.
Of course, deepchecks can easily be integrated with many additional tools, here you can find 
examples and code snippets for inspiration. Contributions to this docs section are very welcome!

.. toctree::
    :maxdepth: 2
    :caption: Integrations

    integrations/spark_databricks
    integrations/pytest
    integrations/h2o
    integrations/hugging_face
    integrations/airflow
    integrations/cml
    integrations/junit
    general/exporting_results/examples/plot_exports_output_to_wandb