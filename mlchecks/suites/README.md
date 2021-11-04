# Check Suites

## Using Existing CheckSuites

### Prebuilt Suites

[**Integrity Suites**](./integrity_suite.py)

  - SingleDatasetIntegrityCheckSuite
  - ComparativeIntegrityCheckSuite
  - IntegrityCheckSuite

[**Leakage Suites**](./leakage_suite.py)
  - IndexLeakageCheckSuite - for datasets with an index column
  - DateLeakageCheckSuite - for datasets with a date column
  - DataLeakageCheckSuite  - for all datasets
  - LeakageCheckSuite - containing all three suites above

[**Overfit Suite**](./overfit_suite.py)
  - OverfitCheckSuite


[**Performance Suite**](./performance_suite.py)
  - PerformanceCheckSuite

### Running a Suite
to run a suite, firs import it:
```python
from mlchecks.suites import *
```
And run it with the required datasets:
```python
OverfitCheckSuite().run(model=my_classification_model, train_dataset=ds_train, validation_dataset=ds_val)
```

## Creating Your Custom CheckSuite

Import Checksuites and Checks
```python
from mlchecks import CheckSuite
from mlchecks.checks import *
```
Build the suite with custom checks and desired parameters
```python
MyModelSuite = CheckSuite('Simple Suite For Model Performance',
    ModelInfo(),
    PerformanceReport(),
    TrainValidationDifferenceOverfit(),
    ConfusionMatrixReport(),
    NaiveComparision(),
    NaiveComparision(native_model_type='statistical')
)
```
and run with required parameters (datasets and models)
```python
MyModelSuite.run(model=rf_clf, train_dataset=ds_train, validation_dataset=ds_val, check_datasets_policy='both')
```