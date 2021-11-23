# Check Suites

## Using Existing CheckSuites

### List of Prebuilt Suites

[**Overall Suites**](./overall_suite.py)

  - overall_check_suite - run all deepchecks checks, including checks for index and date
  - overall_classification_check_suite - run all deepchecks checks for classification tasks with no index or date
  - overall_regression_check_suite - run all deepchecks checks for regression tasks with no index or date
  - overall_generic_check_suite - run all deepchecks checks that work regardless of task type with no index or date

[**Integrity Suites**](./integrity_suite.py)

  - single_dataset_integrity_check_suite - for a single dataset / dataframe
  - comparative_integrity_check_suite - comparing two datasets / dataframes
  - integrity_check_suite - includes both check types 

[**Leakage Suites**](./leakage_suite.py)
  - index_leakage_check_suite - for datasets with an index column
  - date_leakage_check_suite - for datasets with a date column
  - data_leakage_check_suite  - for all datasets
  - leakage_check_suite - containing all three suites above

[**Overfit Suite**](./overfit_suite.py)
  - overfit_check_suite
  
[**Performance Suite**](./performance_suite.py)
  - performance_check_suite - run all performance checks
  - classification_check_suite - check performance for classification tasks
  - regression_check_suite - check performance for regression tasks
  - generic_performance_check_suite - check performance for any task type

### Running a Suite
to run a suite, first import it

```python
from deepchecks.suites import *
```
Then run it with the required input parameters (datasets and models)
```python
overfit_check_suite().run(model=my_classification_model, train_dataset=ds_train, test_dataset=ds_test)
```

## Creating Your Custom CheckSuite

Import CheckSuite and Checks from deepchecks

```python
from deepchecks import CheckSuite
from deepchecks.checks import *
```
Build the suite with custom checks and desired parameters
```python
MyModelSuite = CheckSuite('Simple Suite For Model Performance',
    ModelInfo(),
    PerformanceReport(),
    TrainTestDifferenceOverfit(),
    ConfusionMatrixReport(),
    NaiveModelComparision(),
    NaiveModelComparision(naive_model_type='statistical')
)
```
Then run with required input parameters (datasets and models)
```python
MyModelSuite.run(model=rf_clf, train_dataset=ds_train, test_dataset=ds_test, check_datasets_policy='both')
```