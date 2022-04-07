============
Installation
============

Deepchecks requires Python 3 and can be installed using pip or conda, depending on the package manager
that you're working with for most of your packages.

As a best practice we recommend working on `a virtual environment`_ for pip
and with `a conda environment`_ for conda.

.. _a conda environment:
   https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands

.. _a virtual environment:
    https://docs.python.org/3/library/venv.html



Deepchecks For Tabular Data
============================


Local Installation With Pip
-----------------------------

The deepchecks package can be installed from `PyPi <https://pypi.org/project/deepchecks/>`__ using the following command:

.. code-block:: bash

    pip install deepchecks --upgrade

.. note::
    Deepchecks is in active development, which means that new versions are released on a weekly basis and new features are frequently added.
    If you experience any unexpected behavior from deepchecks, the first step to troubleshoot is to upgrade to the latest version.


Local Installation With Conda
--------------------------------

To install the deepchecks package from the conda package manager run

.. code-block:: bash

    conda install -c conda-forge deepchecks

or, if you already have deepchecks installed and want to update then run

.. code-block:: bash

    conda update -c conda-forge deepchecks


Installing Within a Jupyter Notebook
--------------------------------------

Simply run the following command in a notebook cell

.. code-block:: bash

    import sys
    !{sys.executable} -m pip install deepchecks --quiet --upgrade # --user



Deepchecks For Computer Vision
===============================

.. note:: 
   Deepchecks' Computer Vision subpackage is in **beta** release, and is available from PyPi, 
   use at your own discretion. `Github Issues <https://github.com/deepchecks/deepchecks/issues>`_ are
   highly encouraged for feature requests and bug reports.

Istallation of deepchecks for CV should be stated explicitly and it includes
both the installation of the tabular version and of the computer vision subpackage.
Example commands from above should be altered to install `deepchecks[vision]`.


Using Pip
---------

.. code-block:: bash

    pip install "deepchecks[vision]" --upgrade



Start Working with the Package
=================================

Now it's time to :doc:`check out </index>` deepchecks!


Anonymous Usage Telemetry
-------------------------
We want to keep building and improving deepchecks, so we'd like to know how many people are using it.
For that we added an anonymous usage metrics collection system to the package, that sends a single anonymous event on
the first import of the package.

No credentials, data, personal information or anything private is collected (and will never be).

By default, the anonymous collection mechanism is turned on. You can opt-out at any time by setting the
``DISABLE_DEEPCHECKS_ANONYMOUS_TRACKING`` environment variable to ``True``.

.. code-block:: bash

    export DISABLE_DEEPCHECKS_ANONYMOUS_TRACKING=True