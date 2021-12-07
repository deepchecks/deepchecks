# Check Suites

## Using Existing CheckSuites

### List of Prebuilt Suites

[**Overall Suites**](./overall_suite.py)

  - overall_suite - run all deepchecks checks, including checks for index and date
  - overall_classification_suite - run all deepchecks checks for classification tasks with no index or date
  - overall_regression_suite - run all deepchecks checks for regression tasks with no index or date
  - overall_generic_suite - run all deepchecks checks that work regardless of task type with no index or date

[**Distribution Suites**](./distribution_suite.py)

  - data_distribution_suite - run all data distribution checks

[**Integrity Suites**](./integrity_suite.py)

  - single_dataset_integrity_suite - for a single dataset / dataframe
  - comparative_integrity_suite - comparing two datasets / dataframes
  - integrity_suite - includes both check types 

[**Methodology Suites**](./methodology_suite.py)
  - index_leakage_suite - for datasets with an index column
  - date_leakage_suite - for datasets with a date column
  - data_leakage_suite  - for all datasets
  - leakage_suite - containing all three suites above
  - overfit_suite - run all overfit checks
  - methodological_flaws_suite - checks for all methodological flaws, including unused features
  
[**Performance Suite**](./performance_suite.py)
  - performance_suite - run all performance checks
  - classification_suite - check performance for classification tasks
  - regression_suite - check performance for regression tasks
  - generic_performance_suite - check performance for any task type

### Running a Suite
to run a suite, first import it

```python
from deepchecks.suites import *
```
Then run it with the required input parameters (datasets and models)
```python
overfit_suite().run(model=my_classification_model, train_dataset=ds_train, test_dataset=ds_test)
```

## Creating Your Custom Suite

Import Suite and Checks from deepchecks

```python
from deepchecks import Suite
from deepchecks.checks import *
```
Build the suite with custom checks and desired parameters
```python
MyModelSuite = Suite('Simple Suite For Model Performance',
    ModelInfo(),
    PerformanceReport(),
    TrainTestDifferenceOverfit(),
    ConfusionMatrixReport(),
    SimpleModelComparision(),
    SimpleModelComparision(simple_model_type='statistical')
)
```
Then run with required input parameters (datasets and models)
```python
MyModelSuite.run(model=rf_clf, train_dataset=ds_train, test_dataset=ds_test, check_datasets_policy='both')
```