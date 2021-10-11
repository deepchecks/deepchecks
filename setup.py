from setuptools import setup

setup(
    name='mlchecks',
    version='0.0.1',
    packages=['mlchecks'],
    install_requires=[
        'requests',
        'importlib; python_version == "2.6"',
    ],
)
