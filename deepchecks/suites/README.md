<!--
  ~ ----------------------------------------------------------------------------
  ~ Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
  ~
  ~ This file is part of Deepchecks.
  ~ Deepchecks is distributed under the terms of the GNU Affero General
  ~ Public License (version 3 or later).
  ~ You should have received a copy of the GNU Affero General Public License
  ~ along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
  ~ ----------------------------------------------------------------------------
  ~
-->
# Check Suites

## Using Existing CheckSuites

### [List of Prebuilt Suites](./default_suites.py)

  - single_dataset_integrity - Runs a set of checks that are meant to detect integrity issues within a single dataset.
  - train_test_leakage - Runs a set of checks that are meant to detect data leakage from the training dataset to the test dataset.
  - train_test_validation - Runs a set of checks that are meant to validate correctness of train-test split, including integrity, drift and leakage.
  - model_evaluation - Runs a set of checks that are meant to test model performance and overfit.
  - full_suite - Runs all previously mentioned suites and overview checks.
  

### Running a Suite
to run a suite, first import it

```python
from deepchecks.suites import *
```
Then run it with the required input parameters (datasets and models)
```python
model_evaluation().run(model=my_classification_model, train_dataset=ds_train, test_dataset=ds_test)
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