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
# Deepchecks - Test Suites for ML Models and Data

![pyVersions](https://img.shields.io/pypi/pyversions/deepchecks)
![pkgVersion](https://img.shields.io/pypi/v/deepchecks)
![build](https://github.com/deepchecks/deepchecks/actions/workflows/build.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/deepchecks/badge/?version=latest)](https://docs.deepchecks.com/en/latest/?badge=latest)

Deepchecks is a Python package for validating your machine learning models and data,
during the research and development phase. 

With only one line of code you find data integrity problems, distribution mismatches,
and efficiently evaluate your models to find potential vulnerabilities.

## Installation

### Using pip
```bash
pip install deepchecks #--upgrade --user
```
### Using conda
```bash
conda install deepchecks
```

### From source
To clone the repository and do
an [editable install](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs),
run: 
```bash
git clone https://github.com/deepchecks/deepchecks.git
cd deepchecks
pip install -e .
```
## Are You Ready  to Start Checking?

For the full value from Deepchecks' checking suites, we recommend working with:

-   A model compatible with scikit-learn API that you wish to validate (e.g. RandomForest, XGBoost)
    
-   The model's training data with labels
    
-   Test data (on which the model wasn’t trained) with labels  

Of course, in various valiadation phases (e.g. when validating a dataset's integrity,
or examining distributions between a train-test split), not all of the above are required.
Accordingly, many of the checks and some of the suites need only a subset of the above to run.

## Key Concepts

### Check
Each check enables you to inspect a specific aspect of your data and models.
They are the basic building block of the deepchecks package, covering all kinds of common issues,
such as: PerformanceOverfit, DataSampleLeakage, SingleFeatureContribution,
DataDuplicates, and [many more checks](examples/checks).
Each check can have two types of results:
1. A visual result meant for display (e.g. a figure or a table).
2. A return value that can be used for validating the expected check results
   (validations are typically done by adding a "condition" to the check, as explained below).

### Condition
A condition is a function that can be added to a Check, which returns a pass &#x2713;, fail &#x2716;
or warning &#x0021; result, intended for validating the Check's return value. An example for adding a condition would be:
```python
from deepchecks.checks import BoostingOverfit
BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(threshold=0.05)
```
which will fail if there is a difference of more than 5% between the best score achieved on the test set during
the boosting iterations and the score achieved in the last iteration (the model's "original" score on the test set).

### Suite
An ordered collection of checks, that can have conditions added to them.
The Suite enables displaying a concluding report for all of the Checks that ran.
[Here](deepchecks/suites) you can find the [predefined existing suites](deepchecks/suites) and a code example demonstrating how to build
your own custom suite. The existing suites include default conditions added for most of the checks.
You can edit the preconfigured suites or build a suite of your own with a collection of checks and optional conditions.

<p align="center">
   <img src="docs/images/diagram.svg">
</p>

## Usage Examples

### Running a Check
For running a specific check on your pandas DataFrame, all you need to do is:

```python
from deepchecks.checks import TrainTestFeatureDrift
import pandas as pd

train_df = pd.read_csv('train_data.csv')
train_df = pd.read_csv('test_data.csv')
# Initialize and run desired check
TrainTestFeatureDrift().run(train_data, test_data)
```
Which will product output of the type:
><h4>Train Test Drift</h4>
> <p>The Drift score is a measure for the difference between two distributions,
> in this check - the test and train distributions. <br>
> The check shows the drift score and distributions for the features,
> sorted by feature importance and showing only the top 5 features, according to feature importance.
> If available, the plot titles also show the feature importance (FI) rank.</p>
> <p align="left">
>   <img src="docs/images/train-test-drift-output.png">
> </p>

### Running a Suite
Let's take the "iris" dataset as an example:
```python
from sklearn.datasets import load_iris
iris_df = load_iris(return_X_y=False, as_frame=True)['frame']
```
To run an existing suite all you need to do is import the suite and run it -

```python
from deepchecks.suites import single_dataset_integrity
suite = single_dataset_integrity()
suite.run(iris_df)
```
Which will result in printing the summary of the check conditions and then the visual outputs of all of the checks that
are in that suite.

For a full suite demonstration, check out the [**Quickstart Notebook**](https://docs.deepchecks.com/en/stable/examples/howto-guides/quickstart_in_5_minutes.html).

### Documentation
- HTML documentation (stable release): <https://docs.deepchecks.com/>
- HTML documentation (latest release): <https://docs.deepchecks.com/en/latest>

## Community
- Join our [Slack Community](https://join.slack.com/t/deepcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg) to connect with the maintainers and follow users and interesting discussions
- Post a [Github Issue](https://github.com/deepchecks/deepchecks/issues) to suggest improvements, open an issue, or share feedback.

[comment]: <> "- Send us an [email](mailto:info@deepchecks.com) at info@deepchecks.com"
