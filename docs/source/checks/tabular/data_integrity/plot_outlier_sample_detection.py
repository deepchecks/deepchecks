# -*- coding: utf-8 -*-
"""
.. _plot_tabular_outlier_sample_detection:

Outlier Sample Detection
************************

This notebook provides an overview for using and understanding the Outlier Sample Detection check.

**Structure:**

* `How deepchecks detects outliers <#How-deepchecks-detects-outliers>`__
* `Prepare data <#prepare-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

How deepchecks detects outliers
===============================

Outlier Sample Detection searches for outliers samples (jointly across all features) using the LoOP algorithm.
The LoOP algorithm is a robust method for detecting outliers in a dataset across multiple variables by comparing
the density in the area of a sample with the densities in the areas of its nearest neighbors
(see the `LoOp paper <https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf>`__ for further details).

LoOP relies on a distance matrix. In our implementation we use the Gower distance that averages the distances
per feature between samples. For numeric features it calculates the absolute distance divided by the range of the
feature and for categorical features it is an indicator for whether the values are the same
(see `link <https://www.jstor.org/stable/2528823>`__ for further details).
"""

# %%
# Imports
# =======

import pandas as pd
from sklearn.datasets import load_iris

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import OutlierSampleDetection

# %%
# Prepare data
# ============

iris = pd.DataFrame(load_iris().data)
iris.describe()

# %%
# Add an outlier:

outlier_sample = [1, 10, 50, 100]
iris.loc[len(iris.index)] = outlier_sample
print(iris.tail())
modified_iris = Dataset(iris, cat_features=[])

# %%
# Run the Check
# =============
# We define the nearest_neighbors_percent and the extent parameters for the LoOP algorithm.

check = OutlierSampleDetection(nearest_neighbors_percent=0.01, extent_parameter=3)
check.run(modified_iris)

# %%
# Define a condition
# ==================
# Now, we define a condition that enforces that the ratio of outlier samples in out dataset is below 0.001.

check = OutlierSampleDetection()
check.add_condition_outlier_ratio_less_or_equal(max_outliers_ratio=0.001, outlier_score_threshold=0.9)
check.run(modified_iris)