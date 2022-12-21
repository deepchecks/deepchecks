# -*- coding: utf-8 -*-
"""
Performance Disparity Report
*************************

This notebook provides an overview for using and understanding the Performance Disparity Report check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Automatically detecting weak segments <#automatically-detecting-weak-segments>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
==================================

The check is designed to help you identify subgroups for which the model has a much lower performance score than its baseline score (its overall performance). The subgroups are defined by a chosen *protected* feature (e.g., "sex", "race") and you can specify a *control* feature (e.g., "education") by which to group the data before computing performance differences.

Large performance disparities can indicate a problem with the model. The training data may not be sufficient for certain subgroups or may contain biases, or the model may need to be re-calibrated when applied to certain subgroups. When using appropriate scoring functions, looking at performance disparities can help uncover issues of these kinds.

Remember that this check relies on labeled data provided in the dataset. As such, it can only assess performance disparities to the extent that the labeled data is accurate and representative of the population of interest. Using scoring functions that are robust to class imbalance or that are computed for each model class can help mitigate this issue.
"""

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
# ``control_feature``: Column to use to split the data by groups prior to computing performance disparities.
# 
# ``scorer``: Scoring function to measure performance.
# 
# ``max_segments``: Maximum number of segments into which numerical features are binned. This is also a
# maximum on the number of levels for a categorical ``control_feature``.
#
# ``min_subgroup_size``: Minimum size of a subgroup for which to compute a performance score.
# 
# ``max_subgroups_per_category_to_display``: Maximum number of subgroups (per ``control_feature`` category)
# to display.
# 
# ``max_categories_to_display``: Maximum number of ``control_feature`` levels to display.
# 
# ``use_avg_defaults``: If no scorer was provided, determines whether to return an average score (if True) 
# or a score per class (if False).
# 
# see :class:`API reference <deepchecks.tabular.checks.model_evaluation.PerformanceDisparityReport>` for 
# more details.

from deepchecks.tabular.checks.model_evaluation import PerformanceDisparityReport

check = PerformanceDisparityReport(
   protected_feature="race",
   control_feature="education",
   scorer="f1",
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

result.value.head(3)

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
