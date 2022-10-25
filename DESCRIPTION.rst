
|build| |pkgVersion| |pyVersions|
|Maintainability| |Coverage Status|

..  image:: https://raw.githubusercontent.com/deepchecks/deepchecks/main/docs/source/_static/images/general/deepchecks-logo-with-white-wide-back.png
    :target: https://github.com/deepchecks/deepchecks

Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort.
This includes checks related to various types of issues, such as model performance, data integrity,
distribution mismatches, and more.

What Do You Need in Order to Start Validating?
----------------------------------------------

Depending on your phase and what you wise to validate, you'll need a subset of the following:

- Raw data (before pre-processing such as OHE, string processing, etc.), with optional labels
- The model's training data with labels
- Test data (which the model isn't exposed to) with labels
- A model compatible with scikit-learn API that you wish to validate (e.g. RandomForest, XGBoost)

Deepchecks validation accompanies you from the initial phase when you
have only raw data, through the data splits, and to the final stage of
having a trained model that you wish to evaluate. Accordingly, each
phase requires different assets for the validation. See more about
typical usage scenarios and the built-in suites in the
`docs <https://docs.deepchecks.com/?utm_source=pypi.org&utm_medium=referral&utm_campaign=readme>`__.

Installation
------------

Using pip
~~~~~~~~~

.. code:: bash

    pip install deepchecks #--upgrade --user

Using conda
~~~~~~~~~~~

.. code:: bash

    conda install -c deepchecks deepchecks

.. |build| image:: https://github.com/deepchecks/deepchecks/actions/workflows/build.yml/badge.svg
.. |pkgVersion| image:: https://img.shields.io/pypi/v/deepchecks
.. |pyVersions| image:: https://img.shields.io/pypi/pyversions/deepchecks
.. |Maintainability| image:: https://api.codeclimate.com/v1/badges/970b11794144139975fa/maintainability
   :target: https://codeclimate.com/github/deepchecks/deepchecks/maintainability
.. |Coverage Status| image:: https://coveralls.io/repos/github/deepchecks/deepchecks/badge.svg?branch=main
   :target: https://coveralls.io/github/deepchecks/deepchecks?branch=main
