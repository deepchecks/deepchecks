# -*- coding: utf-8 -*-
"""
.. _nlp__property_drift:

NLP Property Drift
******************

This notebooks provides an overview for using and understanding the nlp property drift check.

**Structure:**

* `Calculating Drift for Text Data <#calculating-drift-for-text-data>`__
* `Prepare data <#prepare-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__
* `Check Parameters <#check-parameters>`__

Calculating Drift for Text Data
=================================

What is Drift?
----------------

Drift is simply a change in the distribution of data over time,
and it is also one of the top reasons why machine learning model's performance degrades over time.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide>`.

How Deepchecks Detects Drift in NLP Data
-----------------------------------------

This check detects drift by in NLP Data by calculated
:ref:`univariate drift measures <drift_detection_by_univariate_measure>` for each of the
:doc:`text property </user-guide/nlp/nlp_properties>` (such as text length, language etc.) that are present in the
train and test datasets.

This check is easy to run (once the properties are calculated once per dataset) and is useful for detecting easily
explainable changes in the data. For example, if you have started to use new data sources that contain
samples in a new language, this check will detect it and show you a high drift score for the language property.

Which NLP Properties Are Used?
-------------------------------

By default the checks use the built-in text properties, and it's also possible to replace the default properties
with custom ones. For the list of the built-in text properties and explanation about custom properties refer to
:doc:`NLP properties </user-guide/nlp/nlp_properties>`.

Prepare data
=============
"""

from deepchecks.nlp.datasets.classification.tweet_emotion import load_data

train_dataset, test_dataset = load_data()

# # Calculate properties, commented out because it takes a short while to run
# train_dataset.calculate_default_properties(include_long_calculation_properties=True)
# test_dataset.calculate_default_properties(include_long_calculation_properties=True)

#%%
# Run the check
# =============

from deepchecks.nlp.checks import PropertyDrift
check_result = PropertyDrift().run(train_dataset, test_dataset)
check_result

#%%
# We can see that there isn't any significant drift in the data. We can see some slight increase in the formality of
# the text samples in the test dataset.
#
# To display the results in an IDE like PyCharm, you can use the following code:

#  check_result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# Observe the check’s output
# --------------------------
# The result value is a dict that contains drift score and method used for each text property.

check_result.value

#%%
# Define a condition
# ==================
# We can define a condition that make sure that nlp properties drift scores do not
# exceed allowed threshold.

check_result = (
    PropertyDrift()
    .add_condition_drift_score_less_than(0.001)
    .run(train_dataset, test_dataset)
)
check_result.show(show_additional_outputs=False)

#%%
# Check Parameters
# ==================
#
# The Property Drift Check can define a list of properties to use for the drift check, or a list to exclude using the
# ``properties`` and ``ignore_properties`` parameters.
#
# On top of that the Property Drift Check supports several parameters pertaining to the way drift is calculated and
# displayed. Information about the most relevant of them can be found
# in the :doc:`drift guide </user-guide/general/drift_guide>`.
