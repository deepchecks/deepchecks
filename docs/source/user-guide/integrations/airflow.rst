Airflow
=======

.. note::
    Download the full code example from the following
    `link <https://github.com/deepchecks/deepchecks/tree/main/examples/integrations/airflow>`__

Apache Airflow is an open-source workflow management system. It is commonly used to automate data processing,
data science, and data engineering pipelines.

This tutorial demonstrates how deepchecks can be used with Apache Airflow. We will run a simple Airflow DAG that will
evaluate the Adult dataset from the UCI Machine Learning Repository. The DAG will run the
:func:`~deepchecks.tabular.suites.data_integrity` and the :func:`~deepchecks.tabular.suites.model_evaluation`
suites on the Adult data and a pre-trained model.

.. image:: /_static/images/integrations/airflow_dag.png
   :alt: The DAG for this tutorial
   :align: center

.. literalinclude:: ../../../../examples/integrations/airflow/deepchecks_airflow_tutorial.py
    :language: python
    :lines: 1-13
    :tab-width: 0

Defining the Data & Model Loading Tasks
---------------------------------------

.. literalinclude:: ../../../../examples/integrations/airflow/deepchecks_airflow_tutorial.py
    :language: python
    :lines: 16-39
    :tab-width: 0

.. note::
    The dataset and the model are saved in the local filesystem for simplicity. For most use-cases,
    it is recommended to save the data and the model in a S3/GCS/other intermediate storage.

Defining the Integrity Report Task
----------------------------------

The :func:`~deepchecks.tabular.suites.data_integrity` suite will be used to evaluate the train and production
datasets. It will check for integrity issues and will save the output html reports to the ``suite_results`` directory.

.. literalinclude:: ../../../../examples/integrations/airflow/deepchecks_airflow_tutorial.py
    :language: python
    :lines: 42-63
    :tab-width: 0

Defining the Model Evaluation Task
----------------------------------

The :func:`~deepchecks.tabular.suites.model_evaluation` suite will be used to evaluate the model itself.
It will check for model performance and overfit issues and will save the report to the ``suite_results`` directory.

.. literalinclude:: ../../../../examples/integrations/airflow/deepchecks_airflow_tutorial.py
    :language: python
    :lines: 66-80
    :tab-width: 0

Creating the DAG
----------------
After we have defined all the tasks, we can create the DAG using Airflow syntax. We will define a DAG that will run
every day.

.. literalinclude:: ../../../../examples/integrations/airflow/deepchecks_airflow_tutorial.py
    :language: python
    :lines: 83-115
    :tab-width: 0


And that's it! In order to run the dag, make sure you place the file in your DAGs folder referenced in your
``airflow.cfg``. The default location for your DAGs is ``~/airflow/dags``.

The DAG is scheduled to run daily, but the scheduling can be configured using the ``schedule_interval`` property.
The DAG can also be manually triggered for a single run by using the following command:

.. code-block:: bash

    airflow dags backfill deepchecks_airflow_integration --start-date <some date in YYYY-MM-DD format>
