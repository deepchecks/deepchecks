
# MLChecks

![pyVersions](https://img.shields.io/pypi/pyversions/mlchecks)
![pkgVersion](https://img.shields.io/pypi/v/mlchecks)
![build](https://github.com/deepchecks/mlchecks/actions/workflows/build.yml/badge.svg)
![coverage](https://deepchecks-public.s3.eu-west-1.amazonaws.com/mlchecks/coverage.svg)
![pylint](https://deepchecks-public.s3.eu-west-1.amazonaws.com/mlchecks/pylint.svg)

MLChecks is a Python package, for quickly and efficiently validating many aspects of your trained machine learning models. These include model performance related issues, machine learning methodology best-practices, model and data integrity.

## Key Concepts

#### Check
Each check enables you to inspect a specific aspect of your data and models. They are the basic building block of the MLChecks package, covering all kinds of common issues, such as:

-   PerformanceOverfit
    
-   DataSampleLeakage
    
-   SingleFeatureContribution
    
-   DataDuplicates
    
-   … and [many more](./notebooks)
    

Each check displays a visual result and returns a custom result value that can be used to validate the expected check results by setting conditions upon them.

#### Suite
An ordered collection of checks. [Here](./mlchecks/suites) you can find the existing suites and a code example for how to add your own custom suite. You can edit the preconfigured suites or build a suite of your own with a collection of checks and result conditions.
  

## Installation

#### Using pip with package wheel file
From the directory in which you have the wheel file, run:
```bash
pip3 install MLChecks-latest.whl #--user
```

#### From source
First clone the repository and then install the package from inside the repository's directory:
```bash
git clone https://github.com/deepchecks/MLChecks.git
cd MLChecks
python setup.py install # --user
```


## Are You Ready  to Start Checking?

To discover the full value from MLChecking your data and model, we recommend having in your jupyter environment:

-   A scikit-learn API supporting model that you wish to validate
    
-   The models’ training data with labels
    
-   Validation data (on which the model wasn’t trained) with labels  

Additionally, many of the checks and some of the suites need only a subset of the above to run.

## Usage Examples

### Running a Check
For running a specific check on your dataframe, all you need to do is:
```python
from mlchecks.checks import *
import pandas as pd

df_to_check = pd.read_csv('data_to_validate.csv')
# Initialize and run desired check
RareFormatDetection().run(df_to_check)
```
Which might product output of the type:
><h4>Rare Format Detection</h4>
> <p>Check whether columns have common formats (e.g. \"XX-XX-XXXX\" for dates\") and detects values that don't match.</p>
> <p><b>&#x2713;</b> Nothing found</p>

If all was fine, or alternatively something like:
><h4>Rare Format Detection</h4>
><p>Check whether columns have common formats (e.g. \"XX-XX-XXXX\" for dates\") and detects values that don't match.</p>
>
>
> Column date:
> <table border="1" class="dataframe" style="text-align: left;">
>   <thead>
>     <tr>
>       <th class="blank level0" >&nbsp;</th>
>       <th class="col_heading level0 col0" >digits and letters format (case sensitive)</th>
>     </tr>
>   </thead>
>   <tbody>
>     <tr>
>       <th id="T_ae5e3_level0_row0" class="row_heading level0 row0" >ratio of rare samples</th>
>       <td id="T_ae5e3_row0_col0" class="data row0 col0" >1.50% (3)</td>
>     </tr>
>     <tr>
>       <th id="T_ae5e3_level0_row1" class="row_heading level0 row1" >common formats</th>
>       <td id="T_ae5e3_row1_col0" class="data row1 col0" >['2020-00-00']</td>
>     </tr>
>     <tr>
>       <th id="T_ae5e3_level0_row2" class="row_heading level0 row2" >examples for values in common formats</th>
>       <td id="T_ae5e3_row2_col0" class="data row2 col0" >['2021-11-07']</td>
>     </tr>
>     <tr>
>       <th id="T_ae5e3_level0_row3" class="row_heading level0 row3" >values in rare formats</th>
>       <td id="T_ae5e3_row3_col0" class="data row3 col0" >['2021-Nov-04', '2021-Nov-05', '2021-Nov-06']</td>
>     </tr>
>   </tbody> </table>

If mismatches were detected.

### Running a Suite
Let's take for example the iris dataset:
```python
import pandas as pd
from sklearn.datasets import load_iris
```
```python
iris_df = load_iris(return_X_y=False, as_frame=True)['frame']
train_len = round(0.67*len(iris_df))
df_train = iris_df[:train_len]
df_val = iris_df[train_len:]
```
To run an existing suite all you need to do is import the suite and run it -
```python
from mlchecks.suites import IntegrityCheckSuite
IntegrityCheckSuite.run(train_dataset=df_train, validation_dataset=df_val, check_datasets_policy='both')
```
Which will result in printing the outputs of all of the checks that are in that suite.

### Example Notebooks
For full usage examples, check out: 
- [**MLChecks Quick Start Notebook**](./notebooks/examples/models/Iris%20Dataset%20CheckSuite%20Example.ipynb) - for a simple example notebook for working with checks and suites.
- [**Example Checks Output Notebooks**](./notebooks) - to see all of the existing checks and their usage examples.

## Communication
- Join our [Slack Community](https://join.slack.com/t/mlcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg) to connect with the maintainers and follow users and interesting discussions
- Post a [Github Issue](https://github.com/deepchecks/MLChecks/issues) to suggest improvements, open an issue, or share feedback.

[comment]: <> "- Send us an [email](mailto:info@mlchecks.com) at info@mlchecks.com"