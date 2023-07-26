# -*- coding: utf-8 -*-
"""

.. _nlp__under_annotated_metadata_segments:

Under Annotated Metadata Segments
*********************************

This notebook provides an overview for using and understanding the under annotated metadata segments check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Automatically detecting under annotated segments <#automatically-detecting-under-annotated-segments>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
==================================

The Under-Annotated Metadata Segments check is designed to help you easily identify segments in your data which are
under-annotated compared to the rest of your dataset, based on the provided :ref:`metadata <nlp__metadata_guide>`.
The check could be very useful for example for identifying a specific data source for which there was
less labeled data. The check can be guided to run only on a specific list of metadata columns,
enabling you to focus on columns where you know a problem exists, or on important business segments.

Automatically detecting under annotated segments
================================================

The check contains two main steps:

#. We train multiple simple tree based models, each one is trained using exactly two
   metadata columns (out of the ones selected above) to predict whether a sample will have a label.

#. We extract the corresponding data samples for each of the leaves in each of the trees (data segments) and calculate
   the annotation ratio in the samples within in. We keep the segments with the lowest annotation ratio.
"""
#%%
# Generate data & model
# =====================

from deepchecks.nlp.utils.test_utils import load_modified_tweet_text_data

text_data = load_modified_tweet_text_data()
text_data.metadata.head(3)

#%%
# Run the check
# =============
#
# The check has several key parameters (that are all optional) that affect the behavior of the
# check and especially its output.
#
# ``columns / ignore_columns``: Controls which columns should be searched for under annotated segments. By default,
# uses all columns.
#
# ``segment_minimum_size_ratio``: Determines the minimum size of segments that are of interest. The check will
# return data segments that contain at least this fraction of the total data samples. It is recommended to
# try different configurations
# of this parameter as larger segments can be of interest even the model performance on them is superior.
#
# ``categorical_aggregation_threshold``: By default the check will combine rare categories into a single category called
# "Other". This parameter determines the frequency threshold for categories to be mapped into to the "other" category.
#
# ``multiple_segments_per_column``: If True, will allow the same metadata column to be a segmenting feature in multiple
# segments, otherwise each metadata column can appear in one segment at most. True by default.
#
# see :class:`API reference <deepchecks.tabular.checks.model_evaluation.WeakSegmentsPerformance>` for more details.

from deepchecks.nlp.checks import UnderAnnotatedMetaDataSegments

check = UnderAnnotatedMetaDataSegments(segment_minimum_size_ratio=0.07,
                                       multiple_segments_per_column=True)
result = check.run(text_data)
result.show()

#%%
# Observe the check's output
# --------------------------
#
# We see in the results that the check indeed found several under annotated segments.
# In the scatter plot display we can see the under annotated segment as well as the annotation distribution with
# respect to the two metadata columns that are relevant to the segment. In order to get the full list
# of under annotated segments found we will inspect
# the ``result.value`` attribute. Shown below are the 3 segments with the worst performance.


result.value['weak_segments_list'].head(3)

#%%
# Define a condition
# ==================
#
# We can add a condition that will validate the annotation ratio in all data segment is
# above a certain threshold.
# A scenario where this can be useful is when we want to make sure that we have enough annotations for quality
# evaluation of the model or drift on a subset of the data that is of interest to us,
# for example for specific age or gender groups.

# Let's add a condition and re-run the check:

check.add_condition_segments_annotation_ratio_greater_than(0.7)
result = check.run(text_data)
result.show(show_additional_outputs=False)
