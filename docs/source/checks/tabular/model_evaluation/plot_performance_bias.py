# -*- coding: utf-8 -*-
"""
Performance Bias
****************

This notebook provides an overview for using and understanding the Performance Bias check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
==================================

The check is designed to help you identify subgroups for which the model has a much lower performance score than its
baseline score (its overall performance). The subgroups are defined by a chosen *protected* feature (e.g., "sex",
"race") and you can specify a *control* feature (e.g., "education") by which to group the data before computing
performance differences. This is primarily useful for fairness analyses, but can also be used to identify other types
of performance disparities.

Large performance disparities can indicate a problem with the model. The training data may not be sufficient for
certain subgroups or may contain biases, or the model may need to be re-calibrated when applied to certain subgroups.
When using appropriate scoring functions, looking at performance disparities can help uncover issues of these kinds.

Remember that this check relies on labeled data provided in the dataset. As such, it can only assess performance
disparities to the extent that the labeled data is accurate and representative of the population of interest. Using
scoring functions that are robust to class imbalance or that are computed for each model class can help mitigate this
issue. """

#%%
# Generate data & model
# =====================

from deepchecks.tabular.datasets.classification.adult import (
    load_data, load_fitted_model)

train_dataset, test_dataset = load_data()
model = load_fitted_model()

#%%
# Run the check
# =============
#
# The check requires the argument ``protected_feature`` identifying a column that defines the subgroups
# for which performance disparities are assessed. In addition, the check has several optional parameters
# that affect its behavior and output.
# 
# * ``control_feature``: Column to use to split the data by groups prior to computing performance disparities.
# 
# * ``scorer``: Scoring function to measure performance. Default to "accuracy" for classification tasks 
#   and "r2" for regression tasks.
# 
# * ``max_subgroups_per_control_cat_to_display``: Maximum number of subgroups (per ``control_feature`` category)
#   to display.
# 
# * ``max_control_cat_to_display``: Maximum number of ``control_feature`` categories to display.
# 
# see :class:`API reference <deepchecks.tabular.checks.model_evaluation.PerformanceBias>` for
# more details.

from deepchecks.tabular.checks.model_evaluation import PerformanceBias

check = PerformanceBias(
   protected_feature="race",
   control_feature="education",
   scorer="accuracy",
   max_segments=3)
result = check.run(test_dataset, model)
result.show()

#%%
# Observe the check's output
# --------------------------
# 
# We see in the results that the check identified the largest performance disparity for the subgroup 
# "Others" within the category of "HS-grad" for the control feature "education". The model performance 
# on this subgroup is 0.095 versus 0.258 for this entire education category.

result.value['scores_df'].head(3)

#%%
# Define a condition
# ==================
#
# We can define on our check a condition that will validate all performance disparities fall within
# a certain threshold. If the condition is not met, the check will fail.
# 
# Let's add a condition and re-run the check:

check.add_condition_bounded_performance_difference(lower_bound=-0.1)
result = check.run(test_dataset, model)
result.show(show_additional_outputs=False)

# %%
