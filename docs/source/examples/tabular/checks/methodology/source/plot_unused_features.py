# -*- coding: utf-8 -*-
"""
Unused Features
***************
This notebook provides an overview for using and understanding the Unused Features check.

**Structure:**

* `How unused features affects my model? <#how-unused-features-affects-my-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

How unused features affects my model?
=====================================
Having too many features can lengthen training times and cause inaccuracies in the model, and is known as "The Curse
of Dimensionality" or "Hughes Phenomenon". This is because the more dimensions there are, the larger the sample space,
which can cause samples to be further apart from each other, as they are different in more dimensions, which badly
affect distance measures on the samples. It might lead the model to overfit the data since it won't be able to
generalize well when all the samples are far from each other.

When finding features with low model contribution, they are probably a statistical noise without any real significance,
meaning they are increasing the data dimensionality without any benefit, and they should be considered for removal.


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

UnusedFeatures().add_condition_number_of_high_variance_unused_features_not_greater_than()
result = UnusedFeatures().run(train_ds, test_ds, model)
result

# %%
# Controlling the variance threshold
# ----------------------------------
# The check can be configured to use a different threshold which controls which features are considered "high variance".
# The default value is `0.4`. We will use a more strict value and see that are fewer features that considered "high
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
check = UnusedFeatures().add_condition_number_of_high_variance_unused_features_not_greater_than(5)
result = check.run(train_ds, test_ds, model)
result.show(show_additional_outputs=False)
