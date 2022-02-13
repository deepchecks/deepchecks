.. raw:: html

   <!--
     ~ ----------------------------------------------------------------------------
     ~ Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
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
Vision Suites
======

Using an Existing Suite
=========================

List of Prebuilt Suites
---------------------------

-  train_test_validation - Runs a set of checks that are meant to
   validate correctness of train-test split, including integrity, drift
   and leakage.
-  model_evaluation - Runs a set of checks that are meant to test model
   performance and overfit.
-  full_suite - Runs all previously mentioned suites and overview
   checks.

Running a Suite
----------------

To run a prebuilt suite, first import it

.. code:: python

   from deepchecks.vision.suites import *

Then run it with the required (suite-dependant) input parameters

.. code:: python

   model_evaluation().run(model=my_classification_model, train_dataset=ds_train, test_dataset=ds_test)

Creating Your Custom Suite
============================

Import Suite and Checks from deepchecks

.. code:: python

   from deepchecks.vision import Suite
   from deepchecks.vision.checks import *

Build the suite with custom checks and desired parameters

.. code:: python

   from ignite.contrib.metrics import ROC_AUC
   MyModelSuite = Suite('Suite with AUC performance',
       ClassPerformance(alternative_metrics=[ROC_AUC]),
       TrainTestLabelDrift()
   )

Then run with required input parameters (datasets and models)

.. code:: python

   MyModelSuite.run(model=rf_clf, train_dataset=ds_train, test_dataset=ds_test)
