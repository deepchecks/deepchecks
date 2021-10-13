import setuptools
from setuptools import setup
from distutils.util import convert_path
import os

main_ns = {}
ver_path = convert_path('mlchecks/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

requirementPath = os.path.dirname(os.path.realpath(__file__)) + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='mlchecks',
    version=main_ns['__version__'],
    packages=['mlchecks'],
    install_requires=install_requires
)
