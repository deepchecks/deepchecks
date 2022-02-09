================
Installation
================

Deepchecks requires Python 3 and can be installed using pip or conda, depending on the package manager you're working with for most of your packages.

Local Installation
====================

Using Pip
-----------
As a best practice we recommend working on a `virtual environment <https://docs.python.org/3/library/venv.html>`__. 

The deepchecks package can be installed from `PyPi <https://pypi.org/project/deepchecks/>`__ using the following command:

.. code-block:: bash

    pip install deepchecks --upgrade

.. note::
    Deepchecks is in alpha development, which means that new versions are released on a weekly basis and new features are frequently added.
    If you experience any unexpected behavior from deepchecks, the first step to troubleshoot is to upgrade to the latest version.
     

Using Conda
------------
As a best practice we recommend `creating a conda environment`_.

.. _creating a conda environment:
   https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands

To install the deepchecks package from the conda package manager run

.. code-block:: bash

    conda install -c conda-forge deepchecks

or, if you already have deepchecks installed and want to update then run

.. code-block:: bash

    conda update -c conda-forge deepchecks

Installing On Google Colab or on a Kaggle Kernel
==================================================

Simply run the following command in a notebook cell

.. code-block:: bash

    !pip install deepchecks --user



Start Working with the Package
=================================
Now it's time to :doc:`get started </index>` with deepchecks!