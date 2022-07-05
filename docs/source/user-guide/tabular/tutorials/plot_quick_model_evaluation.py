# -*- coding: utf-8 -*-
"""
.. _quick_model_evaluation:

Quickstart - Model Evaluation Suite (Wine Quality Data)
************************************************************

The deepchecks model evaluation suite is relevant any time you wish to
evaluate your model on a given test set. For example:

- Analysing the model performance after the training procedure has been completed
- Comparing different models
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
# Load and Prepare Data
# ====================================================
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
# From the result we can retrieve a number of interesting insights, first lets inspect the "Didn't Pass" section.
# From the Performance report check we can see that our model over fitted the training data resulting in
# reduced performance on the test set. From the Regression Systematic Error check we can see that our model tends to
# in average to predict a higher quality than the actual and from the Weak Segments Performance check we can see that
# there are some specific sub-spaces that are not well represented by the model. Examples for those sub-spaces are
# wines with low total sulfur dioxide and wines with high alcohol percentage. When training to optimize
# our model we might want to put special attention to those sub-spaces (e.g. via increased weights to those sub spaces).
#
# Next, let us examine the "Passed" section. From the Simple Model Comparison check we can see that our model
# performs better than the baseline model, an opposite result can indicate a possible problem with the model training
# procedure or the data in was trained on. From the Boosting Overfit and the Unused Features checks we can see that
# model has a well calibrating boosting stopping rule and that it make good use on the different data features.
#
# Let us try and fix the over fitting problem found in the model.
#
# Fix the Model and Re-run the Model Evaluation Suite
# ^^^^^^^^^^

gbr = GradientBoostingRegressor(n_estimators=20)
gbr.fit(X_train, y_train)
suite_result = evaluation_suite.run(train_ds, test_ds, gbr)
suite_result

#%%
#
# We mitigated the over fitting to some extant, yet we increased the model prediction bias. Additional model tuning
# is still required however for now we will remove the relevant conditions from the suite.
#
# Updating an Existing Suite
# ----------------------
#
# To create our own suite, we can start with an empty suite and add checks and condition to it, or we can start with
# one of the default suite and update it accordingly.
#
# let's inspect the suite's structure
evaluation_suite

#%%
#
# Next, we will update the Performance Report condition and remove the Regression Systematic Error check:

evaluation_suite[0].clean_conditions()
evaluation_suite[0].add_condition_train_test_relative_degradation_not_greater_than(0.25)
evaluation_suite.remove(7)

#%%
#
# Now we can re-run the suite using:

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
