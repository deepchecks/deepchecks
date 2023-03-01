"""
.. _vision_simple_classification_tutorial:

==================================
Image Data Validation in 5 Minutes
==================================

Deepchecks Vision is built to validate your data and model, however complex your model and data may be. That
being said, sometime there is no need to write a full-blown
:doc:`classification task </user-guide/vision/auto_tutorials/plot_classification_tutorial>`,
:doc:`object detection task </user-guide/vision/auto_tutorials/plot_detection_tutorial>` or
:doc:`semantic segmentation task </user-guide/vision/auto_tutorials/plot_segmentation_tutorial>`. In the case of a simple classification
task, there are quite a few checks that can be run writing only a few lines of code. In this tutorial, we will show you
how to run all checks that do not require a model on a simple classification task.

This is ideal, for example, when receiving a new dataset for a classification task. Running these checks on the dataset
before even starting with training will give you a quick idea of how the dataset looks like and what potential issues
it contains.

.. code-block:: bash

    # Before we start, if you don't have deepchecks vision package installed yet, run:
    import sys
    !{sys.executable} -m pip install "deepchecks[vision]" --quiet --upgrade # --user

    # or install using pip from your python environment
"""

#%%
# Downloading the Data
# ====================
#
# For this example we'll use a small sample of the RGB `EuroSAT dataset <https://github.com/phelber/eurosat#>`_.
# EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes
# with 27000 labeled and geo-referenced samples.
#
# Citations:
#
# [1] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick
# Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth
# Observations and Remote Sensing, 2019.
#
# [2] Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.
# Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium,
# 2018.

import urllib.request
import zipfile

import numpy as np

url = 'https://figshare.com/ndownloader/files/34912884'
urllib.request.urlretrieve(url, 'EuroSAT_data.zip')

with zipfile.ZipFile('EuroSAT_data.zip', 'r') as zip_ref:
    zip_ref.extractall('EuroSAT')

#%%
# Loading a Simple Classification Dataset
# ========================================
#
# A simple classification dataset is an image dataset structured in the following way:
#
#         - root/
#             - train/
#                 - class1/
#                     image1.jpeg
#             - test/
#                 - class1/
#                     image1.jpeg

# Note that the following function requires torchvision>=0.11.3 to be installed
from deepchecks.vision import classification_dataset_from_directory

train_ds, test_ds = classification_dataset_from_directory(
    root='./EuroSAT/euroSAT/', object_type='VisionData', image_extension='jpg')

#%%
# Running Deepchecks' ``train_test_validation`` suite
# ====================================================
# That's it, we have just defined the classification data object and are ready can run the different deepchecks suites
# and checks. Here we will demonstrate how to run train_test_validation suite:
#
# for additional information on the different suites and checks available see our
# :doc:`Vision Checks </checks_gallery/vision>` gallery.

from deepchecks.vision.suites import train_test_validation

suite = train_test_validation()
result = suite.run(train_ds, test_ds)

#%%
# Observing the Results
# ======================
# The results can be saved as an HTML file with the following code:

result.save_as_html('output.html')

#%%
# Or, if working inside a notebook, the output can be displayed directly by simply printing the result object:

result.show()

#%%
# Understanding the Results
# ===========================
# Looking at the results we see one check whose condition has failed: Feature Label Correlation.
#
# The :doc:`Property Label Correlation Change
# </checks_gallery/vision/train_test_validation/plot_property_label_correlation_change>` check computes various
# :doc:`image properties </user-guide/vision/vision_properties>` and checks if the image label can be inferred using a
# simple model (for example, a Classification Tree) using the property values. The ability to predict the label using
# these properties is measured by the Predictive Power Score (PPS) and this measure is compared between the training
# and test dataset. In this case, the condition alerts us to the fact that the PPS for the "RMS Contrast" property was
# significantly higher in the training dataset than in the test dataset.
#
# We'll show the relevant plot again for ease of discussion:

check_idx = np.where([result.results[i].check.name() == 'Property Label Correlation Change'
                      for i in range(len(result.results))])[0][0]
result.results[check_idx]

#%%
# Here we can see the plot dedicated to the PPS of the property RMS Contrast, which measures the contrast in the
# image by calculating the grayscale standard deviation of the image. This plot shows us that specifically for the
# classes "Forest" and "SeaLake" (the same culprits from the Similar Image Leakage condition), the contrast is a
# great predictor, but only in the training data! This means we have a critical problem - or model may learn to
# classify these classes using only the contrast, without actually learning anything about the image content. We can
# now go on and fix this issue (perhaps by adding train augmentations, or enriching our training set), before we
# have even trained a model on this task.
