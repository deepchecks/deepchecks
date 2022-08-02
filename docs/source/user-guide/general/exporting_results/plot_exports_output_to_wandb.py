# -*- coding: utf-8 -*-
"""
Weights & Biases (wandb)
************************

In this guide, we will demonstrate how to export the Check's and the Suite's output to wandb.

This enables to view the exported results in wandb and compare the results you
receive with different parameters or model/data.

**Structure:**

* `Load Data <#load-data>`__
* `Run a Check <#run-a-check>`__
* `Export Check to wandb <#export-a-check-s-output-checkresult-to-wandb>`__
* `Export Suite to wandb <#exporting-a-suite-s-output-suiteresult-to-wandb>`__
"""

#%%
# Load Data
# ---------

from deepchecks.tabular.datasets.classification import iris

train_dataset, test_dataset = iris.load_data()
model = iris.load_fitted_model()

#%%
# Run a Check
# -----------

from deepchecks.tabular.checks import MultivariateDrift

result = MultivariateDrift().add_condition_overall_drift_value_less_than(
).run(train_dataset, test_dataset, model)

#%%
# Observe CheckResult Display and Value
# -------------------------------------

result

#%%
# the value can be observed
result.value

#%%
# Export a Check's Output (CheckResult) to wandb
# ==============================================
# Exporting the output to wandb is possible using the ``to_wandb`` function. This function
# exports the check outputs to a wandb project.
#
# The output display that will be export will be a bit different from what you usually
# see. Only the tables and the plots are being exported.
#
# On default if you export a single check to wandb without a wandb run active it will
# create a project with the name deepchecks and the and the check's metadata in the
# config and export the results there.
#
# See Check to_wandb options
# --------------------------

from deepchecks.core import CheckResult

help(CheckResult.to_wandb)

#%%
# To use this function you first need to login to your wandb account

import wandb

wandb.login()

#%%

result.to_wandb()

#%%
# Exporting a Suite's Output (SuiteResult) to wandb
# =================================================
# Run Suite and export to wandb
# -----------------------------
# Exporting a suite to wandb is the same as exporting a check. The display in wand
# will have a section per check.
#
# On default if you export a suite to wandb without a wandb run active it will create
# a project with the name deepchecks and the suite's name in the config and send the
# results there.

from deepchecks.tabular.suites import full_suite

suite = full_suite()

#%%

suite_result = suite.run(train_dataset=train_dataset, test_dataset=test_dataset, model=model)
suite_result.to_wandb()

#%%
# You can also set all the kwargs the wandb.init will get:

suite_result.to_wandb(project='my-suite-project', config={'suite-name': 'full-suite'})
