# MLChecks - <TODO: oneline explanation>

![pyVersions](https://img.shields.io/pypi/pyversions/mlchecks) 
![pkgVersion](https://img.shields.io/pypi/v/mlchecks) 
![build](https://github.com/deepchecks/mlchecks/actions/workflows/build.yml/badge.svg) 
![coverage](https://deepchecks-public.s3.eu-west-1.amazonaws.com/mlchecks/coverage.svg)
![pylint](https://deepchecks-public.s3.eu-west-1.amazonaws.com/mlchecks/pylint.svg)


**<TODO: a super short paragraph about the pacakge, what it achieves, etc a super small python example to show how easy it is to integrate and check>**

## Features
**<TODO: a list of features you can achieve with the package>**


## Installation

### Dependencies
MLChecks is supported & tested on **Minimum Python Version of 3.6 onward**

### Using pip
MLChecks is on PyPI, so you can use `pip` to install it:

```bash
# Install the latest Version
pip install mlchecks #--user
# Install a specific version
pip install mlchecks==0.0.5 #--user
```

### From source

if you want to install from source, first, clone this repository:
```bash
git clone git@github.com:deepchecks/MLChecks.git
```
cd into the cloned repository using `cd mlchecks`
then, you can run 
```
python setup.py install # --user
```



## Usage

using is **TODO: ADD some nice and cool words that make it looks fun** 
here is a basic example with `iris` dataset from `sklearn` in order to get the mode info:
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
- TODO: link to Documentation Site
- [FAQ](FAQ.md) ? Link to the Documentation FAQ page (if we have it)
- Publications: medium links? 

### Communication
- (slack)?
- (mail)?
- Github Issues: https://github.com/deepchecks/MLChecks/issues
<!--- - Github Discussions: **TODO: add when OpenSource and is added to the repo** --->


## License

[LICENSE_NAME](LICENSE)
