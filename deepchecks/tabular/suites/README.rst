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

======
Suites
======

Using an Existing Suite
=========================

List of Prebuilt Suites
---------------------------

-  dataset_integrity - Runs a set of checks that are meant to
   detect integrity issues within a single dataset.
-  train_test_validation - Runs a set of checks that are meant to
   validate correctness of train-test split, including integrity, drift
   and leakage.
-  model_evaluation - Runs a set of checks that are meant to test model
   performance and overfit.
-  full_suite - Runs all previously mentioned suites and overview
   checks.
-  production_suite - Runs a set of checks selected for testing model
   performance, drift and integrity in production. Exact checks will
   depend on the task type and on whether the suite should compare
   the production data to a reference dataset.

Running a Suite
----------------

To run a prebuilt suite, first import it

.. code:: python

   from deepchecks.tabular.suites import *

Then run it with the required (suite-dependant) input parameters

.. code:: python

   model_evaluation().run(model=my_classification_model, train_dataset=ds_train, test_dataset=ds_test)

Creating Your Custom Suite
============================

Import Suite and Checks from deepchecks

.. code:: python

   from deepchecks.tabular import Suite
   from deepchecks.tabular.checks import *

Build the suite with custom checks and desired parameters

.. code:: python

   MyModelSuite = Suite('Simple Suite For Model Performance',
       ModelInfo(),
       PerformanceReport(),
       TrainTestDifferenceOverfit(),
       ConfusionMatrixReport(),
       SimpleModelComparision(),
       SimpleModelComparision(strategy='most_frequent')
   )

Then run with required input parameters (datasets and models)

.. code:: python

   MyModelSuite.run(model=rf_clf, train_dataset=ds_train, test_dataset=ds_test)
