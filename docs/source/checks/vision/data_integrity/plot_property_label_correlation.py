# -*- coding: utf-8 -*-
"""
.. _plot_vision_feature_label_correlation:

Property Label Correlation
**************************

This notebook provides an overview for using and understanding the "Property Label Correlation" check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================
The check estimates for every image property (such as brightness, contrast etc.)
its ability to predict the label by itself.

This check can help find a potential bias the dataset, that might lead to shortcut learning - the labels being
strongly correlated with simple image properties such as color, brightness,
aspect ratio and more. This is a critical problem, that will likely stay hidden
without this check (as it won't pop up when comparing model performance on train
and test).

A famous example is the case of wolfs vs. dogs classification, where a model needs to classify whether an image
contains a wolf or a dog, and can learn to do it by the background instead of the actual animal - in the dataset
all the wolves were photographed in the snow and therefore had a white background while all the dogs were photographed in the
grass and therefore had a green background.

The check is based on calculating the predictive power score (PPS) of each image
property. For more details you can read here `how the PPS is calculated
<#how-is-the-predictive-power-score-pps-calculated>`__.

What is a problematic result?
-----------------------------

Image properties with a high predictive score - can indicate that there is a bias in the dataset, as a single property
can predict the label successfully, using simple classic ML algorithms.

This means that a deep learning algorithm may accidentally learn these properties instead of more accurate complex
abstractions. For example, in the dataset of wolves and dogs photographs, the brightness of the image may be used to
predict the label "wolf" easily.


How do we calculate for different vision tasks?
-----------------------------------------------

* For classification tasks, this check uses PPS to predict the class by image properties.
* For object detection tasks, this check uses PPS to predict the class of each
  bounding box, by the image properties of that specific bounding box.
  This means that for each image, this check crops all the sub-images defined by bounding
  boxes, and uses them as inputs as though they were regular classification dataset images.

"""
#%%
# Run the Check
# -------------
# In this example we will run the check on the dataset of wolves vs. dogs.
# For example purposes we picked 10 images of dogs and 10 images of wolves out of the full dataset.
# The original data was downloaded from https://www.kaggle.com/datasets/harishvutukuri/dogs-vs-wolves.

from deepchecks.vision.checks import PropertyLabelCorrelation
from deepchecks.vision.simple_classification_data import classification_dataset_from_directory
import albumentations as A
import urllib.request
import zipfile

url = 'https://figshare.com/ndownloader/files/36671001'
urllib.request.urlretrieve(url, 'wolves_vs_dogs_mini.zip')

with zipfile.ZipFile('wolves_vs_dogs_mini.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

dataset = classification_dataset_from_directory(
    'wolves_vs_dogs_mini', object_type='VisionData', transforms=A.Resize(64, 64))

check_result = PropertyLabelCorrelation().run(dataset)
check_result.show()

#%%
# We can see that both the "Brightness" property and the "Mean Green Relative Intensity" property have a significant
# ability to predict the label.
#
# This is as expected - pictures of wolves have higher brightness because they appear with a white background,
# while dogs appear with a green background, making "Green-ness" a strong predictor for an image containing a dog.
# Using this check we can be made aware of these artifacts, and can solve them (for example by collecting images with
# different backgrounds) before training any kind of model.

#%%
# Define a condition
# ------------------
# We can define a condition to verify that the results are less than a certain threshold.
check_result = PropertyLabelCorrelation().add_condition_property_pps_less_than(0.5).run(dataset)
check_result.show(show_additional_outputs=False)

#%%
# We can now see that the condition failed because the results here are above the threshold.
