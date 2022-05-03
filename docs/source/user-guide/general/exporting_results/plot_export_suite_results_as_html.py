# -*- coding: utf-8 -*-
"""
Export Suite Output to a HTML Report
************************************

In this guide, we will demonstrate how to export a suite's output as an
HTML report. This enables easily sharing the results easier and also
using deepchecks outside of the notebook environment.

**Structure:**

* `Save Suite Result to an HTML Report <#save-suite-result-to-an-html-report>`__
"""

#%%
# Load Data
# ---------
# Let's fetch the iris train and test datasets

from deepchecks.tabular.datasets.classification import iris

train_dataset, test_dataset = iris.load_data()

#%% 
# Run Suite
# ---------

from deepchecks.tabular.suites import full_suite

suite = full_suite()

#%%

suite_result = suite.run(train_dataset=train_dataset, test_dataset=test_dataset)

#%%
# Save Suite Result to an HTML Report
# ===================================
# Exporting the suite's output to an HTML file is possible using the ``save_as_html``
# function. This function expects a file-like object, whether it's a file name or
# the full path to the destination folder.

suite_result.save_as_html('my_suite.html')

# or
suite_result.save_as_html() # will save the result in output.html

#%%

# Removing outputs created. this cell should be hidden in nbpshinx using "nbsphinx: hidden" in the metadata
import os

os.remove('output.html')
os.remove('my_suite.html')

#%%
# **Working with in-memory buffers**
#
# The suite output can also be written into a file buffers. This can be done by
# setting the file argument with a ``StringIO`` or ``BytesIO`` buffer object.

import io

html_out = io.StringIO()
suite_result.save_as_html(file=html_out)

#%%
# View Suite Output
# -----------------
# The suite's output can still be viewed within the notebook

suite_result
