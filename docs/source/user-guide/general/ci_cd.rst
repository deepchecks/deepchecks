=================================
Using Deepchecks In CI/CD
=================================

CI/CD is a practice of automating steps in a software life cycle such as testing, and deploying. The same practice of
traditional software development can be applied to machine learning, which allows for frequent and automated testing
and validation of models, which reduce the risk of errors and improve the overall quality of the model.

For example, CI/CD in machine learning can be used in scenarios such as:

* Data integrity checks: Before the model training process, it’s important to validate the integrity of the data used
  for training. This can include checks such as data completeness, missing values, and data type consistency.
* Model training: The model is trained on the validated data set.
* Model evaluation: The trained model is evaluated using test data and various metrics such as accuracy, precision,
  recall, etc.
* Model deployment: The model is deployed to production if it meets the specified criteria.


By automating these steps through a CI/CD engine and utilizing deepchecks package in the pipeline, it ensures that the
model is thoroughly tested and validated before it is deployed to production.

Deepchecks can be used in the CI/CD process at 2 main steps of the model training process:

* Before the model training process, it’s important to validate the integrity of the data used for training, and the
  train-test split
* After the model training process, it’s important to validate the model’s performance.

In this guide we will show an end to end examples of validating both data and the trained model, but in reality
you may want to split the process into 2 separate pipelines, one for data validation and one for model validation.
In addition we will use the default suites which are provided by deepchecks, but you can also create your own
custom suites which fit your needs.

Integrations
============
Deepchecks can be used in any CI/CD platform, in this guide we will show how to integrate with Airflow and GitHub
Actions.

Airflow Integration
-------------------

.. image:: /_static/images/cicd/airflow.png
   :alt: Airflow DAG example
   :align: center

Apache Airflow is an open-source workflow management system. It is commonly used to automate data processing,
data science, and data engineering pipelines.
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
steps on every push to the `main` branch. Note that we might want to stop the pipeline in case of a suite not passing,
in this case we can change the methods return types from `return suite_result.passed()` to:

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
