# -*- coding: utf-8 -*-
"""
.. _plot_vision_robustness_report:

Robustness Report
*****************

This notebooks provides an overview for using and understanding robustness report check.

**Structure:**

* `How Does the RobustnessReport Check Work? <#how-does-the-robustnessreport-check-work>`__
* `What Are Image Augmentations? <#what-are-image-augmentations>`__
* `Check requirements <#check-requirements>`__
* `Generate data and model <#generate-data-and-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

How Does the RobustnessReport Check Work?
===============================================
This check performs augmentations on images in the dataset, and measures the change
in model performance for each augmentation. This is done in order to estimate how well
the model generalizes on the data.

What Are Image Augmentations?
===========================================
Augmentations on images are any transformation done on the image, such as changing brightness and scale.
The are used during model training for 2 reasons:

* Data in training set is limited, and there's a need to give the model more data
  samples to learn on, especially ones with augmentations that don't necessarily
  exist in training dataset but may be encountered in out-of-sample data.
* As the model relearns the same images again and again in each epoch, augmentations
  on data are done in order to force the model to learn a more generalized version
  of the image, so it will not overfit on specific images.

If Performance Decreases Significantly on Augmented Images, This Could Mean That:
---------------------------------------------------------------------------------
* Training dataset was not diverse enough for the model to learn its features in a generalized way.
* Augmentations on train dataset were either not performed, or not done enough.

When Is It Ok That the Model Will Decrease Performance Due to Augmentations?
----------------------------------------------------------------------------
* If out-of-sample data is not expected to be augmented in these ways, it may not be of concern
  that the model's performance decreases. However, this could still mean that the model does not
  generalize well enough, and therefore can decrease in performance for other types of data shift.
* If augmentations are too extreme, the image may be changed without recognition. In this case,
  where the human eye or professional eye cannot perform the needed task as well, it is expected
  that the model will not be able to infer correctly as well.

Check requirements
==================
The augmentations are usually performed in the `Dataset.__getitem__` method, using a
transformations object. In order to run the check we need to be able to add the
augmentations as the first augmentation in the transforms function. Therefore you need to:

1. Define in `VisionData` the name of your transformations field. The default field name is "transforms"
2. Use either `imgaug` or `Albumentations` libraries as the transformations mechanism.
3. For object detection: Use a single transformation object for both the data and
   the labels (use "transforms" instead of "transform" + "target_transform")

Default Augmentations
=====================
================  ===================================
Image Type        Augmentation name
================  ===================================
Grayscale         `RandomBrightnessContrast <https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast>`__
Grayscale         `ShiftScaleRotate <https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ShiftScaleRotate>`__
RGB               `HueSaturationValue <https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HueSaturationValue>`__
RGB               `RGBShift <https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RGBShift>`__
================  ===================================

"""

#%%
# Generate data and model
# -----------------------

from deepchecks.vision.datasets.classification.mnist_torch import (load_dataset,
                                                                   load_model)

mnist_dataloader_test = load_dataset(train=False, batch_size=1000, object_type='VisionData')
model = load_model()

#%%
# Run the check
# -------------

import torch.nn as nn

from deepchecks.vision.checks import RobustnessReport

result = RobustnessReport().run(mnist_dataloader_test, model)
result

#%%
# If you have a GPU, you can speed up this check by passing it as an argument to .run() as device=<your GPU>
#
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# Observe the check’s output
# --------------------------
# As we see in the results, the check applied different augmentations on the input data,
# and then compared the model's performance on the original images vs the augmeneted
# images. We then compare the overall metrics and also the metrics per class, and we can
# see the difference of the worst degraded classes.
#
# As a result value the check returns per augmentation the overall metrics with their
# relative difference from the original metrics.

result.value

#%%
# Define a condition
# ------------------
# We can define a condition that enforce our model's performance is not degrading by more
# than given percentage when the data is augmeneted

check = RobustnessReport().add_condition_degradation_not_greater_than(0.05)
result = check.run(mnist_dataloader_test, model)
result.show(show_additional_outputs=False)