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
<p align="center">
   <img src="docs/images/deepchecks-banner.png">
</p>

<p align="center">
  <a href="https://docs.deepchecks.com">Documentation</a>
  &emsp | &emsp 
  <a href="https://join.slack.com/t/deepcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg">Join Slack</a>
  &emsp | &emsp 
  <a href="https://deepchecks.com/blog/">Blog</a>
  &emsp | &emsp 
  <a href="https://twitter.com/deepchecks">Twitter</a>
</p>

# Deepchecks - Test Suites for ML Models and Data

![pyVersions](https://img.shields.io/pypi/pyversions/deepchecks)
![pkgVersion](https://img.shields.io/pypi/v/deepchecks)
![build](https://github.com/deepchecks/deepchecks/actions/workflows/build.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/deepchecks/badge/?version=latest)](https://docs.deepchecks.com/en/latest/?badge=latest)

Validate your machine learning models and data, during the research and development phase. 

With only one line of code you can discover data integrity problems, distribution mismatches,
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

<p align="center">
   <img src="docs/images/diagram.svg">
</p>

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
Let's take the "iris" dataset as an example
```python
from sklearn.datasets import load_iris
iris_df = load_iris(return_X_y=False, as_frame=True)['frame']
```
To run an existing suite all you need to do is to import the suite and to run it

```python
from deepchecks.suites import single_dataset_integrity
suite = single_dataset_integrity()
suite.run(iris_df)
```
Which will result in printing the suite outputs, starting with a summary of the check conditions
>
> <h1 id="summary_NKMZO">Single Dataset Integrity Suite</h1>
> <p>The suite is composed of various checks such as: Mixed Data Types, Is Single Value, String Mismatch, etc...<br>
>        Each check may contain conditions (which results in 
>    <span style="color: green;display:inline-block">✓</span> /
>    <span style="color: red;display:inline-block">✖</span> /
>    <span style="color: orange;font-weight:bold;display:inline-block">!</span>
>    ),
>        as well as other outputs such as plots or tables.<br>
>        Suites, checks and conditions can all be modified (see tutorial [link]).</p>
>
> <hr style="background-color: black;border: 0 none;color: black;height: 1px;">
>
> <h2>Conditions Summary</h2>
>
> <table id="T_7735f_">
  <thead>
    <tr>
      <th class="col_heading level0 col0">Status</th>
      <th class="col_heading level0 col1">Check</th>
      <th class="col_heading level0 col2">Condition</th>
      <th class="col_heading level0 col3">More Info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_7735f_row0_col0" class="data row0 col0"><div style="color: red;text-align: center">✖</div></td>
      <td id="T_7735f_row0_col1" class="data row0 col1"><a href="#IsSingleValue_NKMZO">Single Value in Column - Test Dataset</a></td>
      <td id="T_7735f_row0_col2" class="data row0 col2">Does not contain only a single value for all columns</td>
      <td id="T_7735f_row0_col3" class="data row0 col3">Columns containing a single value: ['target']</td>
    </tr>
    <tr>
      <td id="T_7735f_row1_col0" class="data row1 col0"><div style="color: orange;text-align: center;font-weight:bold">!</div></td>
      <td id="T_7735f_row1_col1" class="data row1 col1"><a href="#DataDuplicates_NKMZO">Data Duplicates - Test Dataset</a></td>
      <td id="T_7735f_row1_col2" class="data row1 col2">Duplicate data is not greater than 0%</td>
      <td id="T_7735f_row1_col3" class="data row1 col3">Found 2.00% duplicate data</td>
    </tr>
    <tr>
      <td id="T_7735f_row2_col0" class="data row2 col0"><div style="color: green;text-align: center">✓</div></td>
      <td id="T_7735f_row2_col1" class="data row2 col1">Mixed Nulls - Test Dataset</td>
      <td id="T_7735f_row2_col2" class="data row2 col2">Not more than 1 different null types for all columns</td>
      <td id="T_7735f_row2_col3" class="data row2 col3"></td>
    </tr>
    <tr>
      <td id="T_7735f_row3_col0" class="data row3 col0"><div style="color: green;text-align: center">✓</div></td>
      <td id="T_7735f_row3_col1" class="data row3 col1">Mixed Data Types - Test Dataset</td>
      <td id="T_7735f_row3_col2" class="data row3 col2">Rare data types in all columns are either more than 10.00% or less than 1.00% of the data</td>
      <td id="T_7735f_row3_col3" class="data row3 col3"></td>
    </tr>
    <tr>
      <td id="T_7735f_row4_col0" class="data row4 col0"><div style="color: green;text-align: center">✓</div></td>
      <td id="T_7735f_row4_col1" class="data row4 col1">String Mismatch - Test Dataset</td>
      <td id="T_7735f_row4_col2" class="data row4 col2">No string variants for all columns</td>
      <td id="T_7735f_row4_col3" class="data row4 col3"></td>
    </tr>
    <tr>
      <td id="T_7735f_row5_col0" class="data row5 col0"><div style="color: green;text-align: center">✓</div></td>
      <td id="T_7735f_row5_col1" class="data row5 col1">String Length Out Of Bounds - Test Dataset</td>
      <td id="T_7735f_row5_col2" class="data row5 col2">Ratio of outliers not greater than 0% string length outliers for all columns</td>
      <td id="T_7735f_row5_col3" class="data row5 col3"></td>
    </tr>
    <tr>
      <td id="T_7735f_row6_col0" class="data row6 col0"><div style="color: green;text-align: center">✓</div></td>
      <td id="T_7735f_row6_col1" class="data row6 col1">Special Characters - Test Dataset</td>
      <td id="T_7735f_row6_col2" class="data row6 col2">Ratio of entirely special character samples not greater than 0.10% for all columns</td>
      <td id="T_7735f_row6_col3" class="data row6 col3"></td>
    </tr>
  </tbody>
</table>

Followed by the visual outputs of all of the checks that are in that suite, that isn't appended here for brevity.

For a full suite demonstration, check out the
[**Quickstart Notebook**](https://docs.deepchecks.com/en/stable/examples/howto-guides/quickstart_in_5_minutes.html)
and apply it on your own data and models.

### Documentation
- <https://docs.deepchecks.com/> - HTML documentation (stable release)
- <https://docs.deepchecks.com/en/latest> - HTML documentation (latest release)

## Community
- Join our [Slack Community](https://join.slack.com/t/deepcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg) to connect with the maintainers and follow users and interesting discussions
- Post a [Github Issue](https://github.com/deepchecks/deepchecks/issues) to suggest improvements, open an issue, or share feedback.

[comment]: <> "- Send us an [email](mailto:info@deepchecks.com) at info@deepchecks.com"
