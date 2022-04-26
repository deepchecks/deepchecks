# -*- coding: utf-8 -*-
"""
Similar Image Leakage
***************************
This notebook provides an overview for using and understanding the "Similar Image Leakage" check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================
The check helps identify if the training dataset contains any images that are similar to any images in the test dataset.
Such a situation is nearly always a case of leakage, because we can expect that the model will have an easier time
getting correct predictions on an image that is similar to an image in the training set, compared to it's "real world"
performance. This may mean that the metrics we're seeing for the test data are too optimistic, and we should remove
those similar images from the test set.

How is Similarity calculated?
-------------------------------------
The similarity is calculated using an image hash known as Average Hash. This hash compresses the image using the
following algorithm:

#. Resize the image to a very compact form (the check default is 8X8).
#. Compute the average of the image pixels.
#. For each pixel, replace value by the boolean result of `pixel_value >= image_average`.

Now we end up with a representation of the image that is 8 bytes long, but still contains some real information about
the image content.

We then proceed to check for similar images by searching for test images whose hash is close to a hash of a training
image, when distance is defined by the Hamming distance between the binary vectors that are the hashed images.

Note about default parameters
--------------------------------------
Similarity between images is dependent on the purpose of the dataset. This is because sometimes we're training a model
to find large differences between images (e.g. people vs dogs) and sometimes we're training to find small differences
(e.g. different types of trees). Moreover, sometimes our images are taken from real-world datasets, where they were
taken by different people, in different locations, and sometimes the images are "cleaner", such as ones taken under
microscope or from the same security camera with the same background.

The check's default parameters are set to match a real-world rgb photos and their differences.
If your dataset has more delicate differences in it, it is advised to use the *hash_size* and *similarity_threshold*
parameters of this check.
The *hash_size* parameter controls the size of the hashed image. A larger hash_size will enable to find
finer differences will between images (and results in less similarity).
The *similarity_threshold* parameter controls the ratio of pixels that need to be different in order
for 2 images to be considered "different". A lower similarity_threshold will define less images as "similar".

#
# Run the check
# ==============
"""
from deepchecks.vision.checks import SimilarImageLeakage
from deepchecks.vision.datasets.detection.coco import load_dataset


train_ds = load_dataset(train=True, object_type='VisionData', shuffle=False)
test_ds = load_dataset(train=False, object_type='VisionData', shuffle=False)

#%%

check = SimilarImageLeakage()
check.run(train_ds, test_ds)

#%%
# As we can see, no similar images were found.

#%%
# Insert training images into test
# ---------------------------------
# Let's now see what happens when we insert some of the training images into the test set. We'll insert them with some
# changes to brightness to see what happens.

from copy import copy
from PIL import Image
import numpy as np
from deepchecks.vision.utils.test_utils import get_modified_dataloader


test_ds_modified = copy(test_ds)

def get_modification_func():
    other_dataset = train_ds.data_loader.dataset

    def mod_func(orig_dataset, idx):
        if idx in range(5):
            # Run only on the first 5 images
            data, label = other_dataset[idx]
            # Add some brightness by adding 50 to all pixels
            return Image.fromarray(np.clip(np.array(data, dtype=np.uint16) + 50, 0, 255).astype(np.uint8)), label
        else:
            return orig_dataset[idx]

    return mod_func


test_ds_modified._data_loader = get_modified_dataloader(test_ds, get_modification_func())

#%%
# Re-run after introducing the similar images
# --------------------------------------------

check = SimilarImageLeakage()
check.run(train_ds, test_ds_modified)

#%%
# We can see that the check detected the five images from the training set we introduced to the test set.
#
# Define a condition
# ==================
# We can define on our check a condition that will validate no similar images where found. The default is that no
# similar images are allowed at all, but this can be modified as shown here.

check = SimilarImageLeakage().add_condition_similar_images_not_more_than(3)
result = check.run(train_dataset=train_ds, test_dataset=test_ds_modified)
result.show(show_additional_outputs=False)
