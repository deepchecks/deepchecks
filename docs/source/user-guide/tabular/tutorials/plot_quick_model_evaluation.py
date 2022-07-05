# -*- coding: utf-8 -*-
"""
.. _quick_model_evaluation:

Quickstart - Model Evaluation Suite (Wine Quality Data)
************************************************************

The deepchecks model evaluation suite is relevant any time you wish to
evaluate your model on a given test set. For example:

- Through analysis of the model performance after the training procedure has been completed
- Analysing of the model mid-training to revile pitfalls and required fine-tuning
- Checking model performance on a new data batch (with or without comparison to previous data batches)

Here we'll use a wine quality dataset
(:mod:`deepchecks.tabular.datasets.regression.wine_quality`),
to demonstrate how you can run the suite with only a few simple lines of code, 
and see which kind of insights it can find.

.. code-block:: bash

    # Before we start, if you don't have deepchecks installed yet,
    # make sure to run:
    pip install deepchecks -U --quiet #--user
"""

#%%
# Prepare Data and Model
# ======================
#
# Load Data
# -----------

from deepchecks.tabular.datasets.regression import wine_quality

data = wine_quality.load_data(data_format='Dataframe', as_train_test=False)
data.head(2)

#%%
# Split Data and Train a Model
# -----------------------------
#
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['quality'], test_size=0.2)
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

#%%
# Run Deepchecks for Model Evaluation
# ===========================================
#
# Define a Dataset Object
# -------------------------
#
# Create a deepchecks Dataset, including the relevant metadata (label, date, index, etc.).
# Check out :class:`deepchecks.tabular.Dataset` to see all the columns and types
# that can be declared.

from deepchecks.tabular import Dataset

# We state the categorical features,
# otherwise they will be automatically inferred,
# which may be less accurate, therefore stating
# them explicitly is recommended.

# The label can be passed as a column name or
# as a separate pd.Series / pd.DataFrame

# all metadata attributes are optional.
# Some checks require specific attributes and otherwise will not run.

train_ds = Dataset(X_train, label=y_train,cat_features=[])
test_ds = Dataset(X_test, label=y_test,cat_features=[])

#%%
# Run the Deepchecks Suite
# --------------------------
#
# Validate your data with the :class:`deepchecks.tabular.suites.model_evaluation` suite.
# It runs on two datasets and a model, so you can use it to compare the performance of the model between
# any two batches of data (e.g. train data, test data, a new batch of data
# that recently arrived)
#
# Check out the :doc:`"when should you use deepchecks guide" </getting-started/when_should_you_use>`
# for some more info about the existing suites and when to use them.

from deepchecks.tabular.suites import model_evaluation

evaluation_suite = model_evaluation()
suite_result = evaluation_suite.run(train_ds, test_ds, gbr)
# Note: the result can be saved as html using suite_result.save_as_html()
# or exported to json using suite_result.to_json()
suite_result

#%%
# Analyzing the results
# --------------------------
#
# The result showcase a number of interesting insights, first lets inspect the "Didn't Pass" section.
#
# * From the Performance Report check we can deduce that our model overfitted the training data.
# * From the Regression Systematic Error (test set) check we can see that our model has a small positive bias.
# * From the Weak Segments Performance check (test set) we see that there are some specific sub-spaces on which the
#   model performs poorly. Examples for those sub-spaces are
#   wines with low total sulfur dioxide and wines with high alcohol percentage.
#
# Next, let us examine the "Passed" section.
#
# * From the Simple Model Comparison check we see that our model
#   performs better than the baseline models, an opposite result could indicate a problem with the model
#   or the data in was trained on.
# * From the Boosting Overfit and the Unused Features checks we can deduce that
#   model has a well calibrating boosting stopping rule and that it make good use on the different data features.
#
# Let us try and fix the overfitting issue found in the model.
#
# Fix the Model and Re-run the Model Evaluation Suite
# ^^^^^^^^^^

gbr = GradientBoostingRegressor(n_estimators=20)
gbr.fit(X_train, y_train)
suite_result = evaluation_suite.run(train_ds, test_ds, gbr)
suite_result

#%%
#
# We mitigated the overfitting to some extant, yet increased the model prediction bias. Additional model tuning
# is still required however for now we will remove the relevant conditions from the suite.
#
# Updating an Existing Suite
# ----------------------
#
# To create our own suite, we can start with an empty suite and add checks and condition to it
# (see :doc:`/user-guide/general/customizations/examples/plot_create_a_custom_suite`), or we can start with
# one of the default suites and update it as displayed in this section.
#
# let's inspect our model evaluation suite's structure
evaluation_suite

#%%
#
# Next, we will update the Performance Report condition and remove the Regression Systematic Error check:

evaluation_suite[0].clean_conditions()
evaluation_suite[0].add_condition_train_test_relative_degradation_not_greater_than(0.25)
evaluation_suite.remove(7)

#%%
#
# Re-run the suite using:

result = evaluation_suite.run(train_ds, test_ds, gbr)
result.passed(fail_if_warning=False)

#%%
#
# *Note: the check we manipulated will still run as part of the Suite, however
# it won't appear in the Conditions Summary since it no longer has any
# conditions defined on it. You can still see its display results in the
# Additional Outputs section*
#
# For more info about working with conditions, see the detailed
# :doc:`/user-guide/general/customizations/examples/plot_configure_checks_conditions` guide.
