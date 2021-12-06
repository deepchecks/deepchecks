# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import setuptools
from setuptools import setup
from distutils.util import convert_path
import os

main_ns = {}
ver_path = convert_path('deepchecks/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)
VER = main_ns['__version__']

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
    #license='Propietery', #TODO: what is the license
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
