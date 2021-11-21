# Check Suites

## Using Existing CheckSuites

### List of Prebuilt Suites

[**Overall Suites**](./overall_suite.py)

  - OverallCheckSuite - run all deepchecks checks, including checks for index and date
  - OverallClassificationCheckSuite - run all deepchecks checks for classification tasks with no index or date
  - OverallRegressionCheckSuite - run all deepchecks checks for regression tasks with no index or date
  - OverallGenericCheckSuite - run all deepchecks checks that work regardless of task type with no index or date

[**Integrity Suites**](./integrity_suite.py)

  - SingleDatasetIntegrityCheckSuite - for a single dataset / dataframe
  - ComparativeIntegrityCheckSuite - comparing two datasets / dataframes
  - IntegrityCheckSuite - includes both check types 

[**Leakage Suites**](./leakage_suite.py)
  - IndexLeakageCheckSuite - for datasets with an index column
  - DateLeakageCheckSuite - for datasets with a date column
  - DataLeakageCheckSuite  - for all datasets
  - LeakageCheckSuite - containing all three suites above

[**Overfit Suite**](./overfit_suite.py)
  - OverfitCheckSuite
  
[**Performance Suite**](./performance_suite.py)
  - PerformanceCheckSuite - run all performance checks
  - ClassificationCheckSuite - check performance for classification tasks
  - RegressionCheckSuite - check performance for regression tasks
  - GenericPerformanceCheckSuite - check performance for any task type

### Running a Suite
to run a suite, first import it

```python
from deepchecks.suites import *
```
Then run it with the required input parameters (datasets and models)
```python
OverfitCheckSuite.run(model=my_classification_model, train_dataset=ds_train, test_dataset=ds_test)
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