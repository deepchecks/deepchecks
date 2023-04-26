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

This check detects drift by in NLP Data by calculating various properties of the text and writing them into a table,
where each sample has one row per text sample and one column per textual property. Then, drift can be calculated using
:ref:`univariate measures <drift_detection_by_univariate_measure>` on each text property separately.

Which NLP Properties Are Used?
-------------------------------

The properties used by this check are the ones present in datasets prior to running this check. That's why it's
important to first run ``calculate_default_properties`` on both the train and test datasets prior to running this
check. Note also that in order to calculate drift on more complex properties, such as text Toxicity or Fluency,
you need to pass the ``include_long_calculation_properties=True`` parameter to ``calculate_default_properties``. For
more information on how to calculate properties, please visit the :doc:`nlp properties guide
</user-guide/nlp/nlp_properties>`.

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
