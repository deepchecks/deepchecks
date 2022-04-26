"""
================================================
Using Deepchecks Vision With a Few Lines of Code
================================================

Deepchecks Vision is built to validate your data and model, however complex your model and data may be. That
being said, sometime there is no need to write a full-blown
:doc:`ClassificationData </tutorials/vision/examples/plot_classification_tutorial>` or
:doc:`DetectionData </tutorials/vision/examples/plot_detection_tutorial>`. In the case of a simple classification task,
there is quite a few checks that can be run writing only a few lines of code. In this tutorial, we will show you how
to run all checks that do not require a model on a simple classification task.

This is ideal, for example, when receiving a new dataset for a classification task. Running these checks on the dataset
before even starting with training will give you a quick idea of how the dataset looks like and what potential issues
it contains.
"""

#%%
# Defining the data and model
# ===========================
#
# For this example we'll use a small sample of the RGB `EuroSAT dataset <https://github.com/phelber/eurosat#>`_.
# EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes
# with 27000 labeled and geo-referenced samples.

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

from deepchecks.vision.simple_classification_data import load_dataset

train_ds = load_dataset('./EuroSAT/euroSAT/', train=True, object_type='VisionData', image_extension='jpg')
test_ds = load_dataset('./EuroSAT/euroSAT/', train=False, object_type='VisionData', image_extension='jpg')

# Running Deepchecks' full suite
# ==============================
# That's it, we have just defined the classification data object and are ready to run the train_test_validation suite:

from deepchecks.vision.suites import train_test_validation

suite = train_test_validation()
result = suite.run(train_ds, test_ds)

#%%
# Observing the results:
# ===========================
# The results can be saved as a html file with the following code:

result.save_as_html('output.html')

#%%
# Or, if working inside a notebook, the output can be displayed directly by simply printing the result object:

result

#%%
# Understanding the results:
# ===========================
# Looking at the results we see two checks whose conditions have failed:
#
# 1. similar image leakage
# 2. simple Feature Contribution
#
# The first has clearly failed due to the naturally occurring similarity between different ocean / lake image,
# and the prevailing green of some forest images. We may wish to remove some of these duplicate images but for this
# dataset they make sense.
#
# The second failure is more interesting. The :doc:`Simple Feature Contribution
# </examples/vision/checks/source/methodology/plot_simple_feature_contribution>` check computes various :doc:`simple
# image # properties </user-guide/vision/vision_properties>` and checks if the image label can be inferred using a
# simple model (for example, a Classification Tree) using the property values. The ability to predict the label using
# these properties is measures by the Predictive Power Score (PPS) and this measure is compared between the training
# and test dataset. In this case, the condition alerts us to the fact that this PPS for the "RMS Contrast" property was
# significantly higher in the training dataset than in the test dataset.
#
# We'll show the relevant plot again for ease of discussion:

check_idx = np.where([result.results[i].check.name() == 'Simple Feature Contribution'
                      for i in range(len(result.results))])[0][0]
result.results[check_idx].display[0]

#%%
# Here we can see the plot dedicated to the PPS of the property RMS Contrast, which measures the contrast in the
# image by calculating the grayscale standard deviation of the image. This plot shows us that specifically for the
# classes "Forest" and "SeaLake" (the same culprits from the Similar Image Leakage condition), the contrast is a
# great predictor, but only in the training data! This means we have a critical problem - or model may learn to
# classify these classes using only the contrast, without actually learning anything about the image content. We now
# can go on and fix this issue (perhaps by adding train augmentations, or enriching our training set) even before we
# start thinking about what model to train for the task.
