=================================
Using Deepchecks In CI/CD
=================================

This guide will explain the basics of using CI/CD for machine learning, and how deepchecks can be incorporated into
the process.

**Structure:**

* `CI/CD In Machine Learning <#ci/cd-in-machine-learning>`__
* `Airflow Integration <#airflow-integration>`__
* `GitHub Actions Integration <#github-actions-integration>`__

CI/CD In Machine Learning
==========================

CI/CD is a software engineering concept that is used to streamline the process of building, testing and deploying
software products. CI/CD can also be utilized for the ML model lifecycle - to streamline the process of model training
(and retraining), model validation and model deployment.

For example, CI/CD in machine learning can be used in different steps such as:

* Data integrity validation: When the data used for training is collected via automatic processes and pipelines,
  the data may contain errors and problems we haven't encountered before, either due to a bug in the
  data processing pipeline or due to a change in the data source.
  Examples of such problems include:
  :doc:`conflicting labels between similar samples</checks_gallery/tabular/data_integrity/plot_conflicting_labels>`,
  :doc:`high correlation between features</checks_gallery/tabular/data_integrity/plot_feature_feature_correlation>`,
  :doc:`spelling errors in categorical features</checks_gallery/tabular/data_integrity/plot_string_mismatch>`,
  and more.
* Datasets comparison: In many cases it's useful to make sure that there isn't any leakage or drift between 2
  datasets. For example, when doing a time based split of the data there is a risk that the datasets will have
  significant differences, or when doing a periodic model retraining we might want to compare the new dataset
  to the previous one.
  Examples of checks that can be used are:
  :doc:`drift between features</checks_gallery/tabular/train_test_validation/plot_train_test_feature_drift>`,
  :doc:`change in correlation between features and label</checks_gallery/tabular/train_test_validation/plot_feature_label_correlation_change>`,
  :doc:`duplicate samples between the datasets</checks_gallery/tabular/train_test_validation/plot_train_test_samples_mix>`,
  and more.
* Model training: The model is trained on the (previously validated) training set.
* Model validation: The trained model is evaluated on test data, testing for performance, weak segments and more:
  :doc:`performance compared to naive model</checks_gallery/tabular/model_evaluation/plot_simple_model_comparison>`,
  :doc:`calibration score for each class</checks_gallery/tabular/model_evaluation/plot_calibration_score>`,
  etc.
* Model deployment: The model is deployed to production if it meets the specified criteria.

In most cases, the steps above are done manually today by running local tests, and inspecting graphs and reports.
By using CI/CD these time consuming tasks can be automated, freeing up time for more meaningful work.

Deepchecks can be used in the CI/CD process at 2 main steps of model development:

* Before model training, to validate the integrity of the data used for training, and check for any data
  drift or leakage between the train and test datasets.
* After model training, testing model performance across different metrics and data segments, and get
  deeper insights on the model's behavior, such as weak segments and performance bias.

In this guide we will show end to end examples of validating both data and the trained model. In most use cases those
processes will be separated into 2 separate pipelines, one for data validation and one for model validation.
We will use the default suites provided by deepchecks, but it's possible to create a
:doc:`custom suite</user-guide/general/customizations/plot_create_a_custom_suite>`
containing hand chosen checks and
:doc:`conditions</user-guide/general/customizations/plot_configure_check_conditions>`
in order to cater to the specific needs of the project.

Integrations
============
Deepchecks can be used in any CI/CD platform, in this guide we will show how to integrate with Airflow and GitHub
Actions.

Airflow Integration
-------------------

.. image:: /_static/images/cicd/airflow.png
   :alt: Airflow DAG example
   :align: center

Apache Airflow is an open-source workflow management system which is commonly used to automate data processing
pipelines.
In the following example we will use S3 to load the training data and to store our suite results. We define the first
2 tasks as short circuit tasks, which means the rest of the downstream tasks will be skipped if the return value of
them is false. This is useful in cases where we want to stop the pipeline if the data validation failed.
We can also add an additional step of deploying the model after the last validation has passed.

.. literalinclude:: ../../../../examples/cicd/airflow.py
    :language: python
    :tab-width: 0

We can access the result of the pipeline in our S3 bucket:

.. image:: /_static/images/cicd/s3_suites.png
   :alt: Airflow DAG example
   :align: center


GitHub Actions Integration
--------------------------
GitHub Actions is a service that allows you to run automated workflows, which can be triggered by events such as
pushing to a repository or creating a pull request.

We will use the same functions as defined in airflow above with slight changes, and run them in the GitHub Actions
steps on every push to the `main` branch. Note that we might want to stop the pipeline when a suite fails.
In this case we can change the methods return types from `return suite_result.passed()` to:

.. code-block:: python

    if not suite_result.passed():
        sys.exit(1)

The yaml file for the GitHub Actions workflow is as follows:

.. code-block:: yaml

    name: Model Training and Validation

    on:
      push:
        branches: [ main ]
      pull_request:
        branches: [ main ]

    jobs:
      build:
        runs-on: ubuntu-latest
        steps:
          - name: Checkout Repository
            uses: actions/checkout@v3
          - name: Set up Python 3.8
            uses: actions/setup-python@v2
            with:
              python-version: 3.8
          - name: Install dependencies
            run: |
              python -m pip install --upgrade pip
              pip install deepchecks
          - name: Validate Data
            run: |
              python validate_data.py
          - name: Validate Train-Test Split
            run: |
              python validate_train_test_split.py
          - name: Train Model
            run: |
              python train_model.py
          - name: Validate Model Performance
            run: |
              python validate_model_performance.py
          - name: Archive Deepchecks Results
            uses: actions/upload-artifact@v3
            # Always run this step even if previous step failed
            run: always()
            with:
              name: deepchecks results
              path: *_validation.html
