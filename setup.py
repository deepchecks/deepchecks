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
