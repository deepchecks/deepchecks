# MLChecks

![pyVersions](https://img.shields.io/pypi/pyversions/mlchecks) 
![pkgVersion](https://img.shields.io/pypi/v/mlchecks) 
![build](https://github.com/deepchecks/mlchecks/actions/workflows/build.yml/badge.svg) 
![coverage](https://deepchecks-public.s3.eu-west-1.amazonaws.com/mlchecks/coverage.svg)
![pylint](https://deepchecks-public.s3.eu-west-1.amazonaws.com/mlchecks/pylint.svg)


MLChecks is a python package, which provides an easy-to-use method for validating machine learning models. 
Using a single library with a few lines of code, you can perform individual checks and receive visualizations and derived data for your 
performance metrics, data integrity issues, explainability, and other insights.

<!-- toc -->

* [MLChecks](#mlchecks)
  * [More about MLChecks](#more-about-mlchecks)
  * [Installation](#installation)
    + [Dependencies](#dependencies)
    + [Using pip](#using-pip)
    + [From source](#from-source)
  * [Usage](#usage)
  * [Contributing & Development](#contributing---development)
      - [for more information regarding Contribution & Development, see [contributing](.CONTRIBUTING.md)](#for-more-information-regarding-contribution---development--see--contributing--contributingmd-)
  * [Help and Support](#help-and-support)
    + [Documentation](#documentation)
    + [Communication](#communication)
  * [License](#license)
   
<!-- tocstop --> 

## More about MLChecks

MLChecks is a library that contains the following components:

<img src="/Block-Diagram.jpeg" alt="MLChecks - Block diagram"/>

With MLChecks you can achieve the following type of suites:
* Performance report
* Data integrity
* Explainability
* Insights such as drift, confidence metrics, and more others.

Each of these suites allows you to manage a variety of checks with a single line of code:

* [Confusion Matrix report](./notebooks/confusion_matrix_report_example.ipynb)
* [Performance report](./notebooks/performance_report_example.ipynb)
* [ROC report](./notebooks/roc_report_example.ipynb)
* [Index Leakage report]("./notebooks/Index Leakage.ipynb")
* [String mismatch report]("./notebooks/String mismatch.ipynb")
* [Boosting overfit](./notebooks/boosting_overfit.ipynb)
* [Data duplicate](./notebooks/data_duplicats.ipynb)
* [Data Sample leakage](./notebooks/data_sample_leakage.ipynb)
* [Dominant frequency](./notebooks/dominant_frequency_change.ipynb)
* [Mixed Nulls](./notebooks/mixed_nulls.ipynb)
* [Mixed types](./notebooks/mixed_types.ipynb)
* [New Category](./notebooks/new_category.ipynb)
* [Performance overfit](./notebooks/performance_overfit.ipynb)
* [Rare-format detection](./notebooks/rare_format_detection.ipynb)
* [Single feature contribution](./notebooks/single_feature_contribution.ipynb)
* [Special characters](./notebooks/special_characters.ipynb)
* [String mismatch](./notebooks/string_mismatch_comparison.ipynb)


## Installation

### Dependencies

MLChecks is supported & tested on **Python version 3.6 or higher**

### Using pip

MLChecks is on PyPI, so you can use `pip` to install it:

```bash
# Install the latest Version
pip install mlchecks #--user
# Install a specific version
pip install mlchecks==0.0.5 #--user
```

### From Source

if you want to install from source, first, clone this repository:
```bash
git clone git@github.com:deepchecks/MLChecks.git
```
cd into the cloned repository using `cd mlchecks`
then, you can run 
```
python setup.py install # --user
```



## Getting Started

The following code demonstrates how to use MLChecks to query the model info for the `iris` dataset from `sklearn`.
Most of the MLChecks functionality  requires adding just one line to your code.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from mlchecks.checks.overview import model_info

clf = AdaBoostClassifier()
iris = load_iris()
X = iris.data
Y = iris.target
clf.fit(X, Y)

model_info(clf)
```
which will result in the output:


> <table border="1" class="dataframe">   <thead>
>     <tr style="text-align: right;">
>       <th>parameter</th>
>       <th>value</th>
>     </tr>   </thead>   <tbody>
>     <tr>
>       <td>algorithm</td>
>       <td>SAMME.R</td>
>     </tr>
>     <tr>
>       <td>base_estimator</td>
>       <td>None</td>
>     </tr>
>     <tr>
>       <td>learning_rate</td>
>       <td>1.0</td>
>     </tr>
>     <tr>
>       <td>n_estimators</td>
>       <td>50</td>
>     </tr>
>     <tr>
>       <td>random_state</td>
>       <td>None</td>
>     </tr>   </tbody> </table>


**For more Examples take a look at the [notebooks](./notebooks) folder**


## Contributing & Development

We welcome contributors of all experience level, as part of our goal to become the "go-to" package for pre-emptive model validation and checking

on unix based systems, you can use the `makefile` too ease development cycle:
after fetching the source code, it is recommended to run `make all` to run tests and validation


#### for more information regarding Contribution & Development, see [contributing](.CONTRIBUTING.md)



## Help and Support

### Documentation
- [FAQ](FAQ.md) ? Link to the Documentation FAQ page (if we have it)
- Publications: https://deepchecks.com/blog/

### Communication
- Slack: https://join.slack.com/t/mlcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg
- Github Issues: https://github.com/deepchecks/MLChecks/issues
- Github feature request: TODO
<!--- - Github Discussions: **TODO: add when OpenSource and is added to the repo** --->


## License

[LICENSE_NAME](LICENSE)
