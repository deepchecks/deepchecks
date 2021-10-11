from setuptools import setup
import os

requirementPath = os.path.dirname(os.path.realpath(__file__)) + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='mlchecks',
    version='0.0.1',
    packages=['mlchecks'],
    install_requires=install_requires
)
