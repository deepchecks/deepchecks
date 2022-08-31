# -*- coding: utf-8 -*-
"""
.. _quick_model_evaluation:

Quickstart - Model Evaluation Suite
***********************************

The deepchecks model evaluation suite is relevant any time you wish to
evaluate your model. For example:

- Thorough analysis of the model's performance before deploying it.
- Evaluation of a proposed model during the model selection and optimization stage.
- Checking the model's performance on a new batch of data (with or without comparison to previous data batches).

Here we'll build a regression model using the wine quality dataset
(:mod:`deepchecks.tabular.datasets.regression.wine_quality`),
to demonstrate how you can run the suite with only a few simple lines of code, 
and see which kind of insights it can find.

.. code-block:: bash

    # Before we start, if you don't have deepchecks installed yet, run:
    import sys
    !{sys.executable} -m pip install deepchecks -U --quiet

    # or install using pip from your python environment
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
# Split Data and Train a Simple Model
# -----------------------------------
#
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['quality'], test_size=0.2, random_state=42)
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

#%%
# Run Deepchecks for Model Evaluation
# ===========================================
#
# Create a Dataset Object
# -------------------------
#
# Create a deepchecks Dataset, including the relevant metadata (label, date, index, etc.).
# Check out :class:`deepchecks.tabular.Dataset` to see all the column types and attributes
# that can be declared.

from deepchecks.tabular import Dataset

# Categorical features can be heuristically inferred, however we
# recommend to state them explicitly to avoid misclassification.

# Metadata attributes are optional. Some checks will run only if specific attributes are declared.

train_ds = Dataset(X_train, label=y_train, cat_features=[])
test_ds = Dataset(X_test, label=y_test, cat_features=[])

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
suite_result.show()

#%%
# Analyzing the results
# --------------------------
#
# The result showcase a number of interesting insights, first let's inspect the "Didn't Pass" section.
#
# * :doc:`/checks_gallery/tabular/model_evaluation/plot_train_test_performance`
#   check result implies that the model overfitted the training data.
# * :doc:`/checks_gallery/tabular/model_evaluation/plot_regression_systematic_error`
#   (test set) check result demonstrate the model small positive bias.
# * :doc:`/checks_gallery/tabular/model_evaluation/plot_weak_segments_performance`
#   (test set) check result visualize some specific sub-spaces on which the
#   model performs poorly. Examples for those sub-spaces are
#   wines with low total sulfur dioxide and wines with high alcohol percentage.
#
# Next, let's examine the "Passed" section.
#
# * :doc:`/checks_gallery/tabular/model_evaluation/plot_simple_model_comparison` check result states that the model
#   performs better than naive baseline models, an opposite result could indicate a problem with the model
#   or the data it was trained on.
# * :doc:`/checks_gallery/tabular/model_evaluation/plot_boosting_overfit` check
#   and the :doc:`/checks_gallery/tabular/model_evaluation/plot_unused_features` check results implies that the
#   model has a well calibrating boosting stopping rule and that it make good use on the different data features.
#
# Let's try and fix the overfitting issue found in the model.
#
# Fix the Model and Re-run a Single Check
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from deepchecks.tabular.checks import TrainTestPerformance

gbr = GradientBoostingRegressor(n_estimators=20)
gbr.fit(X_train, y_train)
# Initialize the check and add an optional condition
check = TrainTestPerformance().add_condition_train_test_relative_degradation_less_than(0.3)
result = check.run(train_ds, test_ds, gbr)
result.show()

#%%
#
# We mitigated the overfitting to some extent. Additional model tuning is required to overcome
# other issues discussed above. For now, we will update and remove the relevant conditions from the suite.
#
# Updating an Existing Suite
# --------------------------
#
# To create our own suite, we can start with an empty suite and add checks and condition to it
# (see :doc:`/user-guide/general/customizations/examples/plot_create_a_custom_suite`), or we can start with
# one of the default suites and update it as demonstrated in this section.
#
# let's inspect our model evaluation suite's structure
evaluation_suite

#%%
#
# Next, we will update the Train Test Performance condition and remove the Regression Systematic Error check:

evaluation_suite[0].clean_conditions()
evaluation_suite[0].add_condition_train_test_relative_degradation_less_than(0.3)
evaluation_suite = evaluation_suite.remove(7)

#%%
#
# Re-run the suite using:

result = evaluation_suite.run(train_ds, test_ds, gbr)
result.passed(fail_if_warning=False)

#%%
#
# For more info about working with conditions, see the detailed
# :doc:`/user-guide/general/customizations/examples/plot_configure_check_conditions` guide.
