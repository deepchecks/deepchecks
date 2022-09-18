# -*- coding: utf-8 -*-
"""
.. _plot_tabular_segment_performance:

Segment Performance
*******************
"""

#%%
# Load data
# =========
# The dataset is the adult dataset which can be downloaded from the UCI machine learning repository.
#
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
# Irvine, CA: University of California, School of Information and Computer Science.

from deepchecks.tabular.datasets.classification import adult

#%%
# Create Dataset
# ==============

train_ds, validation_ds = adult.load_data()


#%%
# Classification Model
# ====================

model = adult.load_fitted_model()
model

#%%

from deepchecks.tabular.checks import SegmentPerformance

SegmentPerformance(feature_1='workclass', feature_2='hours-per-week').run(validation_ds, model)
