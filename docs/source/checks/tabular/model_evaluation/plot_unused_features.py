# -*- coding: utf-8 -*-
"""
.. _plot_tabular_unused_features:

Unused Features
***************
This notebook provides an overview for using and understanding the Unused Features check.

**Structure:**

* `How unused features affect my model? <#how-unused-features-affect-my-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

How unused features affect my model?
=====================================
Having too many features can prolong training times and degrade model performance due to "The Curse of Dimensionality"
or "Hughes Phenomenon".
This is because the dimensional space grows exponentially with the number of features. When the space is too large in
relate to the number of data samples, it results in a very sparse distribution of the samples in the space.
This sparsity also makes the samples more similar to each other, since they are all far from each other which makes it
harder to find cluster together similar samples in order to find patterns.
The increased dimensional space and samples similarity may require more complex models, which in turn
are in greater risk of overfitting.

Features with low model contribution (feature importance) are probably just noise, and should be removed as they
increase the dimensionality without contributing anything. Nevertheless, models may miss important features. For that
reason the Unused Features check selects out of these features those that have high variance, as they may represent
information that was ignored during model construction. We may wish to manually inspect those features to make sure
our model is not missing on important information.

Run the check
=============
We will run the check on the adult dataset which can be downloaded from the
`UCI machine learning repository <http://archive.ics.uci.edu/ml>`_ and is also available in
`deepchecks.tabular.datasets`.
"""

from deepchecks.tabular.checks import UnusedFeatures
from deepchecks.tabular.datasets.classification import adult

train_ds, test_ds = adult.load_data()
model = adult.load_fitted_model()

UnusedFeatures().add_condition_number_of_high_variance_unused_features_less_or_equal()
result = UnusedFeatures().run(train_ds, test_ds, model)
result

# %%
# Controlling the variance threshold
# ----------------------------------
# The check can be configured to use a different threshold which controls which features are considered "high variance".
# The default value is `0.4`. We will use a more strict value and see that fewer features are considered "high
# variance".
result = UnusedFeatures(feature_variance_threshold=1.5).run(train_ds, test_ds, model)
result

#%%
# Controlling the importance threshold
# ------------------------------------
# We can also define the importance threshold which controls features are considered important. If we define it as 0
# then all features are considered important.
result = UnusedFeatures(feature_importance_threshold=0).run(train_ds, test_ds, model)
result

# %%
# Define a condition
# ==================
# We can define a condition that enforces that number of unused features with high variance is not greater than a given
# amount, the default is 5.
check = UnusedFeatures().add_condition_number_of_high_variance_unused_features_less_or_equal(5)
result = check.run(train_ds, test_ds, model)
result.show(show_additional_outputs=False)
