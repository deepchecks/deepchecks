.. _getting_started:

================
Getting Started
================

Welcome to Deepchecks!

To get started and easily validate your data and models, make sure to
install the deepchecks python package.

Installation
==============

Local Installation
---------------------

Deepchecks requires Python 3 and can be installed using pip or conda, depending on the package manager you're working with for most of your packages.

Using Pip
~~~~~~~~~~
As a best practice we recommend working on a `virtual environment <https://docs.python.org/3/library/venv.html>`__. 

The deepchecks package can be installed from `PyPi <https://pypi.org/project/deepchecks/>`__ using the following command:

.. code-block:: bash

    pip install deepchecks --upgrade

.. note::
    Deepchecks is in alpha development, which means that new versions are released on a weekly basis and new features are frequently added. If you experience any unexpected behavior from deepchecks, the first step to troubleshoot is to upgrade to the latest version.
     

Using Conda
~~~~~~~~~~~~~
As a best practice we recommend `creating a conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`__.

To install the deepchecks package from the conda package manager run

.. code-block:: bash

    conda install -c deepchecks deepchecks

or, if you already have deepchecks installed and want to update then run

.. code-block:: bash

    conda update -c deepchecks deepchecks

Installing On Google Colab
---------------------------
Simply run the following command in the notebook cell:

.. code-block:: bash

    !pip install deepchecks

Installing On a Kaggle Kernel
-------------------------------
Run the following command in the notebook cell:

.. code-block:: bash

    !conda install -c deepchecks deepchecks

Start working with the Package!
=================================
To get started with deepchecks with 5 lines of code, head to :doc:`examples/howto-guides/quickstart_in_5_minutes`.

For additional usage examples and for understanding the best practices of how to use the package, stay tuned for our additional
guides that will be added here with our official launch!