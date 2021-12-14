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
import setuptools
from setuptools import setup
from distutils.util import convert_path
import os

main_ns = {}

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
    description = 'Package for validating your machine learning model and data',
    author = 'deepchecks',  
    author_email = 'info@deepchecks.com', 
    url = 'https://github.com/deepchecks/deepchecks',
    download_url = "https://github.com/deepchecks/deepchecks/releases/download/{0}/deepchecks-{0}.tar.gz".format(VER),
    keywords = ['Software Development', 'Machine Learning'],   
    classifiers         = [
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
