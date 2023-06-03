.. raw:: html

   <!--
     ~ ----------------------------------------------------------------------------
     ~ Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
     ~
     ~ This file is part of Deepchecks.
     ~ Deepchecks is distributed under the terms of the GNU Affero General
     ~ Public License (version 3 or later).
     ~ You should have received a copy of the GNU Affero General Public License
     ~ along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
     ~ ----------------------------------------------------------------------------
     ~
   -->


=======================
Contributing Guidelines
=======================

Pull Request Checklist
======================

-  Read the `contributing
   guidelines <https://github.com/deepchecks/deepchecks/blob/master/CONTRIBUTING.rst>`__
-  Check if your changes are consistent with the
   `guidelines <#general-guidelines-for-contribution>`__
-  Refer to the `Linux/Mac users <#linux-and-mac-users>`__ section if your development environment
   is either Linux or Mac.

   - `Running Unit Tests <#linux-mac-running-unit-tests>`__
   - `Coding Style <#linux-mac-coding-style>`__
   - `Test Coverage <#linux-mac-test-coverage>`__
   - `Generate Docs <#linux-mac-generate-docs>`__
   
-  Refer to the `Windows users <#windows-users>`__ section if your development environment
   is Windows.

   - `Creating Virtual Environment <#creating-virtual-environment>`__
   - `Installing Dependencies <#installing-dependencies>`__
   - `Running Unit Tests <#windows-running-unit-tests>`__
   - `Coding Style <#windows-coding-style>`__

General guidelines for contribution
=====================================

-  Include unit tests when you contribute new features, as they help to:
   
   - Prove that your code works correctly.
   - Guard against future breaking changes to lower the maintenance cost.

-  Bug fixes also generally require unit tests, because the presence of
   bugs usually indicates insufficient test coverage.
-  Make sure that your code changes complies with the coding style of the
   project.
-  Keep API compatibility in mind when you change code. Reviewers of
   your pull request will comment on any API compatibility issues.

Linux and Mac Users
======================
For contributors using either Linux or Mac operating systems, we have created
a ``makefile`` which will help you get all the setup done in just a few commands.
The ``make`` commands will create the virtual environment and install all the
required dependencies on your system.


Running Unit Tests
-------------------

.. _linux-mac-running-unit-tests:

Every Pull Request submitted will be checked on every supported Python
version, in your on-going development, you can run the following command
to verify the unit tests: 

.. code:: bash

   make test

when finishing with your development and prior to creating a pull
request, run the following command in order to run the tests on every
supported python version, thus validating that your PR tests will pass.

.. code:: bash

   make tox

Coding Style
-------------
.. _linux-mac-coding-style:

Changes to Python code should pass both linting and docstring check. You can
run the following command in order to validate your code style using
``pylint`` and ``pydocstring``: 

.. code:: bash

   make validate

Test Coverage
--------------
.. _linux-mac-test-coverage:

To verify whether your changes has affected the test coverage, you can
run the following command: 

.. code:: bash

   make coveralls

Generate Docs
--------------
.. _linux-mac-generate-docs:

To generate the documentation, you can run the following commnad:

.. code:: bash

   make docs

Windows Users
==============
For contributors using Windows operating system, you have to manually run
the following commands since as of now the ``make`` commands works with
Linux/Mac OS. **BTW, If you have a fix for that you are more than welcome to contribute!**


Creating virtual environment
-----------------------------
To create a virtual environment in python, run the following command:

.. code:: bash

   python -m venv <name_of_your_virtual_environment>

Installing dependencies
------------------------
.. _windows-installing-dependencies:

Once your virtual environment is set up, activate your virtual environment
by executing the command: ``./venv/Scripts/Activate.ps1``. Then, install
the dependencies for your virtual environment by running the following command:

- To install regular dependencies:

   .. code:: bash

      pip install -r .\requirements\requirements.txt

- To install development dependencies:

   .. code:: bash

      pip install -r .\requirements\dev-requirements.txt

- To install NLP related dependencies:

   .. code:: bash

      pip install -r .\requirements\nlp-requirements.txt

- To install NLP properties related dependencies:

   .. code:: bash

      pip install -r .\requirements\nlp-prop-requirements.txt

- To install vision development dependencies:

   .. code:: bash

      pip install -r .\requirements\vision-requirements.txt

..

   If you are working with NLP module, then you may require to install
   ``fasttext`` on your machine. To install, you can download a suitable
   version of ``fasttext`` wheel file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext.
   Once downloaded, you can run the following command to install the wheel file:

   .. code:: bash
      
      pip install <name_of_your_wheel_file>.whl

Once you have installed all the dependencies, you are ready to
work on the project.

Running Unit Tests
-------------------
.. _windows-running-unit-tests:

To verify and execute all the unit tests, run the following command:

.. code:: bash

   pytest .\tests

If you want to execute specific tests, execute the commands as follows:

.. code:: bash
   
   # Execute NLP tests 
   pytest .\tests\nlp\

   # Execute vision tests 
   pytest .\tests\vision\
   
   # Execute tabular tests 
   pytest .\tests\tabular\

Coding Style
------------
.. _windows-coding-style:

Changes to Python code should pass both linting and docstring check.
In order to validate your code style, you can run the following commands:

.. code:: bash

   # To run pylint on all the files
   pylint deepchecks

   # To run pylint on specific file
   pylint .\deepchecks\nlp\utils\text_properties.py

   # To run pydocstyle on all the files
   python -m pydocstyle --convention=pep257 --add-ignore=D107 deepchecks
