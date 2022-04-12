# -*- coding: utf-8 -*-
"""
Outlier Sample Detection
***************

This notebooks provides an overview for using and understanding the Outlier Sample Detection check.

**Structure:**

* `How deepchecks detects outliers <#How-deepchecks-detects-outliers>`__
* `Prepare data <#prepare-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

How deepchecks detects outliers
========================
Outlier Sample Detection searches for outliers samples using the LoOP algorithm.
The LoOP algorithm is a robust method for detecting outliers in a dataset across multiple variables by comparing
the density in the area of a sample with the densities in the areas of its nearest neighbors
(see `link <https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf>`_ for further details).

LoOP is build upon a distance matrix, in our implementation we use the Gower distance that averages the distances
per feature between samples. For numeric features it calculates the absolute distance divide by the range of the
feature. For categorical features it is an indicator whether the values are the same
(see `link <https://www.jstor.org/stable/2528823>`_ for further details).
"""

# %%
# Imports
# =======

import pandas as pd
from sklearn.datasets import load_iris

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.integrity.outlier_sample_detection import OutlierSampleDetection

# %%
# Prepare data
# =========

iris = pd.DataFrame(load_iris().data)
iris.describe()

# %%
# Add an outlier:

outlier_sample = [1, 10, 50, 100]
iris.loc[len(iris.index)] = outlier_sample
print(iris.tail())
modified_iris = Dataset(iris)

# %%
# Run the Check
# =============
# We define the num_neighbors and the extent parameters for the LoOP algorithm.

check = OutlierSampleDetection(num_nearest_neighbors=10, extent_parameter=3)
check.run(modified_iris)

# %%
# Define a condition
# ==================
# Now, we define a condition that enforces that the ratio of outlier samples in out dataset is below 0.001.

check = OutlierSampleDetection()
check.add_condition_outlier_ratio_not_greater_than(max_outliers_ratio=0.001, outlier_score_threshold=0.9)
check.run(modified_iris)