"""
.. _vision_segmentation_tutorial:

===============================
Semantic Segmentation Tutorial
===============================

In this tutorial, you will learn how to validate your **semantic segmentation model** using deepchecks test suites.
You can read more about the different checks and suites for computer vision use cases at the
:doc:`examples section  </checks_gallery/vision/index>`.

If you just want to see the output of this tutorial, jump to :ref:`observing_the_result` section.

A semantic segmentation task is a task where every pixel of the image is labeled with a single class.
Therefore, a common output of these tasks is an image of identical size to the input, with a vector for each pixel
of the probability for each class.

.. code-block:: bash

    # Before we start, if you don't have deepchecks vision package installed yet, run:
    import sys
    !{sys.executable} -m pip install "deepchecks[vision]" --quiet --upgrade # --user

    # or install using pip from your python environment
"""

# %%
# Defining the data and model
# ===========================

# Importing the required packages
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F


# %%
# Load Data
# ~~~~~~~~~
# The model in this tutorial is used to detect different objects in images (labels based on the Pascal VOC dataset).
# The model is trained to identify 20 different objects (person, bicycle etc.) and background.
# The dataset itself is the COCO128 dataset with semantic segmentation labels, mapped to the Pascal VOC labels
# (Originally, the COCO dataset includes more labels, but those has been filtered out)
# The dataset can be loaded as a pytorch DataLoader object from deepchecks.vision.datasets.segmentation, as is done in
# this tutorial, but can also be loaded as a SegmentationData object

# The full pascal VOC data and information can be found here: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
# And the COCO128 dataset can be found here: https://www.kaggle.com/datasets/ultralytics/coco128

from torchvision.utils import draw_segmentation_masks

from deepchecks.vision.datasets.segmentation.segmentation_coco import load_dataset, load_model
from deepchecks.vision.segmentation_data import SegmentationData

train_loader = load_dataset(object_type='DataLoader', train=True)
test_loader = load_dataset(object_type='DataLoader', train=False)

# %%
# Visualize a Few Images
# ~~~~~~~~~~~~~~~~~~~~~~
# Let's visualize a few images with their segmentation, to understand the data augmentation.
batch = next(iter(test_loader))

masked_images = [draw_segmentation_masks(batch[0][i], masks=torch.stack([batch[1][i] == j for j in range(20)]),
                                         alpha=0.6) for i in range(5)]

fix, axs = plt.subplots(ncols=len(masked_images), figsize=(20, 20))
for i, img in enumerate(masked_images):
    img = img.detach()
    img = F.to_pil_image(img)
    axs[i].imshow(np.asarray(img))
    axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

fix.show()

#%%
# .. image :: /_static/images/tutorials/segmentation_5_examples.png
#     :alt: COCO images with segmentation labels
#
# Downloading a Pre-trained Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this tutorial, we will download a pre-trained LRSAPP model and a MobileNetV3 Large backbone
# from the official PyTorch repository. For more details, please refer to the
# `official documentation <https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html>`__.


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = load_model(pretrained=True, device=device)

# %%
# Validating the Model With Deepchecks
# =====================================
# Now, after we have the training data, test data and the model, we can validate the model with
# deepchecks test suites.
#
# Visualize the Data Loader and the Model Outputs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# First we'll make sure we are familiar with the data loader and the model outputs.

batch = next(iter(train_loader))

print("Batch type is: ", type(batch))
print("First element is: ", type(batch[0]), "with len of ", len(batch[0]))
print("Example output of an image shape from the dataloader ", batch[0][0].shape)
print("Image values", batch[0][0])
print("-" * 80)

print("Second element is: ", type(batch[1]), "with len of ", len(batch[1]))
print("Example output of a label shape from the dataloader ", batch[1][0].shape)


# %%
# Implementing the SegmentationData class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The checks in the package validate the model & data by calculating various quantities over the data, labels and
# predictions. In order to do that, those must be in a pre-defined format, according to the task type.
# The first step is to implement a class that enables deepchecks to interact with your model and data and transform
# them to this pre-defined format, which is set for each task type.
# In this tutorial, we will implement the semantic segmentation task type by implementing a class that inherits from the
# :class:`deepchecks.vision.segmentation_data.SegmentationData` class.
#
# The SegmentationData class contains additional data and general methods intended for easy access to relevant metadata
# for semantic segmentation ML models validation.
# To learn more about the expected format please visit the API reference for the
# :class:`deepchecks.vision.segmentation_data.SegmentationData` class.


class CocoSegmentationData(SegmentationData):
    """Class for loading the COCO segmentation dataset, inherits from :class:`~deepchecks.vision.SegmentationData`.

    Implement the necessary methods to load the dataset.
    """

    def batch_to_labels(self, batch):
        """Extract from the batch only the labels and return the labels in format (H, W).

        See SegmentationData for more details on format.
        """
        return batch[1]

    def infer_on_batch(self, batch, model, device):
        """Infer on a batch of images and return predictions in format (C, H, W), where C is the class_id dimension.

        See SegmentationData for more details on format.
        """
        normalized_batch = [F.normalize(img.unsqueeze(0).float() / 255,
                                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for img in batch[0]]

        predictions = [model(img)["out"].squeeze(0).detach() for img in normalized_batch]
        predictions = [torch.nn.functional.softmax(pred, dim=0) for pred in predictions]

        return predictions

    def batch_to_images(self, batch):
        """Convert the batch to a list of images, where each image is a 3D numpy array in the format (H, W, C)."""
        return [tensor.numpy().transpose((1, 2, 0)) for tensor in batch[0]]


# %%
# After defining the task class, we can validate it by running the following code:

# The label_map is a dictionary that maps the class id to the class name, for display purposes.
LABEL_MAP = {0: 'background', 1: 'airplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car',
             8: 'cat', 9: 'chair', 10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse', 14: 'motorcycle',
             15: 'person', 16: 'potted plant', 17: 'sheep', 18: 'couch', 19: 'train', 20: 'tv'}

training_data = CocoSegmentationData(data_loader=train_loader, label_map=LABEL_MAP)
test_data = CocoSegmentationData(data_loader=test_loader, label_map=LABEL_MAP)

training_data.validate_format(model, device=device)
test_data.validate_format(model, device=device)

# And observe the output:

# %%
# Running Deepchecks' model evaluation suite on our data and model!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now that we have defined the task class, we can validate the model with the model evaluation suite of deepchecks.
# This can be done with this simple few lines of code:

from deepchecks.vision.suites import model_evaluation

suite = model_evaluation()
result = suite.run(training_data, test_data, model, device=device)

# %%
# .. _vision_segmentation_tutorial__observing_the_result:
#
# Observing the results:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The results can be saved as a html file with the following code:

result.save_as_html('output.html')

# %%
# Or, if working inside a notebook, the output can be displayed directly by simply printing the result object:

result.show()
