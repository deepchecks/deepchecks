# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""

|build| |Documentation Status| |pkgVersion| |pyVersions|
|Maintainability| |Coverage Status|

.. image:: https://raw.githubusercontent.com/deepchecks/deepchecks/main/docs/images/deepchecks-logo-with-white-wide-back.png
  :target: https://github.com/deepchecks/deepchecks

Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort.
This includes checks related to various types of issues, such as model performance, data integrity,
distribution mismatches, and more.

What Do You Need in Order to Start Validating?
----------------------------------------------

Depending on your phase and what you wise to validate, you'll need a
subset of the following:

-  Raw data (before pre-processing such as OHE, string processing,
   etc.), with optional labels

-  The model's training data with labels

-  Test data (which the model isn't exposed to) with labels

-  A model compatible with scikit-learn API that you wish to validate
   (e.g. RandomForest, XGBoost)

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
.. |Documentation Status| image:: https://readthedocs.org/projects/deepchecks/badge/?version=latest
   :target: https://docs.deepchecks.com/en/latest/?badge=latest
.. |pkgVersion| image:: https://img.shields.io/pypi/v/deepchecks
.. |pyVersions| image:: https://img.shields.io/pypi/pyversions/deepchecks
.. |Maintainability| image:: https://api.codeclimate.com/v1/badges/970b11794144139975fa/maintainability
   :target: https://codeclimate.com/github/deepchecks/deepchecks/maintainability
.. |Coverage Status| image:: https://coveralls.io/repos/github/deepchecks/deepchecks/badge.svg?branch=main
   :target: https://coveralls.io/github/deepchecks/deepchecks?branch=main

"""

import setuptools
from setuptools import setup
from distutils.util import convert_path
import os

main_ns = {}
DOCLINES = (__doc__ or '').split("\n")

with open(os.path.join('./', 'VERSION')) as version_file:
    VER = version_file.read().strip()

requirementPath = os.path.dirname(os.path.realpath(__file__)) + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()




setup(
    name='deepchecks',
    version=VER,
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    license_files = ('LICENSE', ),
    description = DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    author = 'deepchecks',  
    author_email = 'info@deepchecks.com', 
    url = 'https://github.com/deepchecks/deepchecks',
    download_url = "https://github.com/deepchecks/deepchecks/releases/download/{0}/deepchecks-{0}.tar.gz".format(VER),
    keywords = ['Software Development', 'Machine Learning'],
    include_package_data=True,
    classifiers         = [
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
