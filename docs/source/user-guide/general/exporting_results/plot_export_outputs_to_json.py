# -*- coding: utf-8 -*-
"""
Export Outputs to JSON
**********************

In this guide, we will demonstrate how to export the Check's and the Suite's output to JSON format.

This enables to use the exported results as a serializeable object that can later be
shared or reviewed in other Python scopes, and also enables working with the check's
output (name, parameters, result value and result display) not only in a visual manner
(e.g. by parsing the output JSON).

**Structure:**

* `Load Data <#load-data>`__
* `Run a Check <#run-a-check>`__
* `Export Check to JSON <#export-a-check-s-output-checkresult-to-a-json-format>`__
* `Export Suite to JSON <#exporting-a-suite-s-output-suiteresult-to-json>`__
"""

#%%
# Load Data
# =========

from deepchecks.tabular.datasets.classification import iris

train_dataset, test_dataset = iris.load_data()
model = iris.load_fitted_model()

#%%
# Run a Check
# ===========

from deepchecks.tabular.checks import MultivariateDrift

result = MultivariateDrift().add_condition_overall_drift_value_less_than(
).run(train_dataset, test_dataset, model)

#%%
# Observe CheckResult Display and Value

result

#%% and value can be observed
result.value

#%%
# Export a Check's Output (CheckResult) to a JSON Format
# ======================================================
# Serialization of the output to a JSON format is possible using the ``to_json`` function.
# This function takes the check outputs and serializes it to a JSON string.
#
# The format of the check's output json includes all info required to reconstruct
# the check run and it's output: it's name, the parameters the check receives, one
# sentence summary of the check's purpose, it's result value and a json of the data
# relevant for displaying the check's outputs.
#
# See Check JSON Structure
# ------------------------

from deepchecks.core import CheckResult

help(CheckResult.to_json)

#%%

# get output JSON
serialized_output = result.to_json()

#%%

import json

# note - conditions_table key exists only if there are conditions that were defined on check
json.loads(serialized_output).keys()

#%%
# Observe JSON Output
# -------------------

json.loads(serialized_output)

#%%
# Display the CheckResult Output from a JSON
# ------------------------------------------
# The serialized JSON can be used to reproduce the run in other Python scopes,
# by using the ``from_json`` function


from deepchecks.utils.json_utils import from_json

from_json(serialized_output)

#%%
# Exporting a Suite's Output (SuiteResult) to JSON
# ================================================
# Run Suite and Save to JSON
# --------------------------

from deepchecks.tabular.suites import full_suite

suite = full_suite()

#%%

suite_result = suite.run(train_dataset=train_dataset, test_dataset=test_dataset, model=model)
suite_json = suite_result.to_json()

#%%
# Observe Suite's JSON Strucutre
# ------------------------------

from deepchecks.core import SuiteResult

help(SuiteResult.to_json)

#%%
# Suite name:

json.loads(suite_json)['name']

#%%
# Results is an array of CheckResult JSON's, let's see how many checks ran in suite:

len(json.loads(suite_json)['results'])

#%%
# Lets observe a specific check's output, which is similar to the `Check output's
# JSON Structure <#export-a-check-s-output-checkresult-to-a-json-format>`__ we saw
# before for each check

json.loads(suite_json)['results'][0]
