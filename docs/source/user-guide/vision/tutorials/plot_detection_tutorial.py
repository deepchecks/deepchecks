"""
.. _vision_detection_tutorial:

==========================
Object Detection Tutorial
==========================

In this tutorial, you will learn how to validate your **object detection model** using deepchecks test suites.
You can read more about the different checks and suites for computer vision use cases at the
:doc:`examples section </checks_gallery/vision/index>`.

If you just want to see the output of this tutorial, jump to the :ref:`observing the results <vision_segmentation_tutorial__observing_the_result>` section.

An object detection tasks usually consist of two parts:

- Object Localization, where the model predicts the location of an object in the image,
- Object Classification, where the model predicts the class of the detected object.

The common output of an object detection model is a list of bounding boxes around the objects, and
their classes.

.. code-block:: bash

    # Before we start, if you don't have deepchecks vision package installed yet, run:
    import sys
    !{sys.executable} -m pip install "deepchecks[vision]" --quiet --upgrade # --user

    # or install using pip from your python environment
"""

#%%
# Defining the data and model
# ===========================
# .. note::
#   In this tutorial, we use the pytorch to create the dataset and model. To see how this can be done using tensorflow
#   or other frameworks, please visit the :ref:`creating VisionData guide <vision_data_class>`.
#
# Load Data
# ~~~~~~~~~
# The model in this tutorial is used to detect tomatoes in images. The model is trained on a dataset consisted of
# 895 images of tomatoes, with bounding box annotations provided in PASCAL VOC format.
# All annotations belong to a single class: tomato.
#
# .. note::
#     The dataset is available at the following link:
#     https://www.kaggle.com/andrewmvd/tomato-detection
#
#     We thank the authors of the dataset for providing the dataset.

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import xml.etree.ElementTree as ET
import urllib.request
import zipfile

url = 'https://figshare.com/ndownloader/files/34488599'
urllib.request.urlretrieve(url, 'tomato-detection.zip')

with zipfile.ZipFile('tomato-detection.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

class TomatoDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.images = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, 'annotations'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")
        bboxes, labels = [], []
        with open(ann_path, 'r') as f:
            root = ET.parse(f).getroot()

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                if int(difficult) == 1:
                    continue
                cls_id = 1
                xmlbox = obj.find('bndbox')
                b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
                        float(xmlbox.find('ymax').text)]
                bboxes.append(b)
                labels.append(cls_id)

        bboxes = torch.as_tensor(np.array(bboxes), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)

        if self.transforms is not None:
            res = self.transforms(image=np.array(img), bboxes=bboxes, class_labels=labels)

        target = {
            'boxes': [torch.Tensor(x) for x in res['bboxes']],
            'labels': res['class_labels']
        }
        img = res['image']

        return img, target

    def __len__(self):
        return len(self.images)

data_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

dataset = TomatoDataset(root=os.path.join(os.path.curdir, 'tomato-detection/data'),
                        transforms=data_transforms)
train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                            [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)],
                                                            generator=torch.Generator().manual_seed(42))
test_dataset.transforms = A.Compose([ToTensorV2()])

#%%
# Visualize the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's see how our data looks like.

print(f'Number of training images: {len(train_dataset)}')
print(f'Number of test images: {len(test_dataset)}')
print(f'Example output of an image shape: {train_dataset[0][0].shape}')
print(f'Example output of a label: {train_dataset[0][1]}')


#%%
# Downloading a Pre-trained Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this tutorial, we will download a pre-trained SSDlite model and a MobileNetV3 Large backbone
# from the official PyTorch repository. For more details, please refer to the
# `official documentation <https://pytorch.org/vision/stable/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html#torchvision.models.detection.ssdlite320_mobilenet_v3_large>`_.
#
# After downloading the model, we will fine-tune it for our particular classes. We will do it by replacing the pre-trained
# head with a new one that matches our needs.
from functools import partial
from torch import nn
import torchvision
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)

in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
num_anchors = model.anchor_generator.num_anchors_per_location()
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
_ = model.to(device)

#%%
# Loading Pre-trained Weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For this tutorial we will not include the training code itself, but will download and load pre-trained weights.

model.load_state_dict(torch.load('tomato-detection/ssd_model.pth'))
_ = model.eval()

#%%
# Validating the Model With Deepchecks
# =====================================
# Now, after we have the training data, test data and the model, we can validate the model with
# deepchecks test suites.
#
# Implementing the VisionData class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The checks in the package validate the model & data by calculating various quantities over the data, labels and
# predictions. In order to do that, those must be in a pre-defined format, according to the task type.
# In the following example we're using pytorch. To see an implementation of this in tensorflow, please refer to
# :ref:`creating VisionData guide <vision_data_class>`
# For pytorch, we will use our DataLoader, but we'll create a new collate function for it, that transforms the batch to
# the correct format. Then, we'll create a :class:`deepchecks.vision.vision_data.vision_data.VisionData` object, that will hold the data loader.
#
# To learn more about the expected format please visit
# :doc:`supported tasks and formats guide </user-guide/vision/supported_tasks_and_formats>`.
#
# First, we will create some functions that transform our batch to the correct format of images, labels and predictions:

def get_untransformed_images(original_images):
    """
    Convert a batch of data to images in the expected format. The expected format is an iterable of images,
    where each image is a numpy array of shape (height, width, channels). The numbers in the array should be in the
    range [0, 255] in a uint8 format.
    """
    inp = torch.stack(list(original_images)).cpu().detach().numpy().transpose((0, 2, 3, 1))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Un-normalize the images
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp * 255

def transform_labels_to_cxywh(original_labels):
    """
    Convert a batch of data to labels in the expected format. The expected format is an iterator of arrays, each array
    corresponding to a sample. Each array element is in a shape of [B, 5], where B is the number of bboxes
    in the image, and each bounding box is in the structure of [class_id, x, y, w, h].
    """
    label = []
    for annotation in original_labels:
        if len(annotation["boxes"]):
            bbox = torch.stack(annotation["boxes"])
            # Convert the Pascal VOC xyxy format to xywh format
            bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]
            # The label shape is [class_id, x, y, w, h]
            label.append(
                torch.concat([torch.stack(annotation["labels"]).reshape((-1, 1)), bbox], dim=1)
            )
        else:
            # If it's an empty image, we need to add an empty label
            label.append(torch.tensor([]))
    return label

def infer_on_images(original_images):
    """
    Returns the predictions for a batch of data. The expected format is an iterator of arrays, each array
    corresponding to a sample. Each array element is in a shape of [B, 6], where B is the number of bboxes in the
    predictions, and each bounding box is in the structure of [x, y, w, h, score, class_id].

    Note that model and device here are global variables, and are defined in the previous code block, as the collate
    function cannot recieve other arguments than the batch.
    """
    nm_thrs = 0.2
    score_thrs = 0.7
    imgs = list(img.to(device) for img in original_images)
    # Getting the predictions of the model on the batch
    with torch.no_grad():
        preds = model(imgs)
    processed_pred = []
    for pred in preds:
        # Performoing non-maximum suppression on the detections
        keep_boxes = torchvision.ops.nms(pred['boxes'], pred['scores'], nm_thrs)
        score_filter = pred['scores'][keep_boxes] > score_thrs

        # get the filtered result
        test_boxes = pred['boxes'][keep_boxes][score_filter].reshape((-1, 4))
        test_boxes[:, 2:] = test_boxes[:, 2:] - test_boxes[:, :2]  # xyxy to xywh
        test_labels = pred['labels'][keep_boxes][score_filter]
        test_scores = pred['scores'][keep_boxes][score_filter]

        processed_pred.append(
            torch.concat([test_boxes, test_scores.reshape((-1, 1)), test_labels.reshape((-1, 1))], dim=1))
    return processed_pred

#%%
# Now we'll create the collate function that will be used by the DataLoader.
# In pytorch, the collate function is used to transform the output batch to any custom format, and we'll use that
# in order to transform the batch to the correct format for the checks.

from deepchecks.vision.vision_data import BatchOutputFormat

def deepchecks_collate_fn(batch) -> BatchOutputFormat:
    """Return a batch of images, labels and predictions in the deepchecks format."""
    # batch received as iterable of tuples of (image, label) and transformed to tuple of iterables of images and labels:
    batch = tuple(zip(*batch))
    images = get_untransformed_images(batch[0])
    labels = transform_labels_to_cxywh(batch[1])
    predictions = infer_on_images(batch[0])
    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)

#%%
# We have a single label here, which is the tomato class
# The label_map is a dictionary that maps the class id to the class name, for display purposes.

LABEL_MAP = {
    1: 'Tomato'
}

#%%
# Now that we have our updated collate function, we can recreate the dataloader in the deepchecks format, and use it
# to create a VisionData object:

from deepchecks.vision.vision_data import VisionData

train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=deepchecks_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=deepchecks_collate_fn)

training_data = VisionData(batch_loader=train_loader, task_type='object_detection', label_map=LABEL_MAP)
test_data = VisionData(batch_loader=test_loader, task_type='object_detection', label_map=LABEL_MAP)

#%%
# Making sure our data is in the correct format:
# ~~~~~~~~~~~~~~~~~~~~~~
# The VisionData object automatically validates your data format and will alert you if there is a problem.
# However, you can also manually view your images and labels to make sure they are in the correct format by using
# the ``head`` function to conveniently visualize your data:

training_data.head()

#%%
# Running Deepchecks' suite on our data and model!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now that we have defined the task class, we can validate the model with the deepchecks' model evaluation suite.
# This can be done with this simple few lines of code:

from deepchecks.vision.suites import model_evaluation

suite = model_evaluation()
result = suite.run(training_data, test_data)

#%%
# We also have suites for:
# :func:`data integrity <deepchecks.vision.suites.data_integrity>`
# - validating a single dataset and
# :func:`train test validation <deepchecks.vision.suites.train_test_validation>` -
# validating the dataset split

#%%
# .. _observing_the_result:
#
# Observing the results:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The results can be saved as a html file with the following code:

result.save_as_html('output.html')

#%%
# Or, if working inside a notebook, the output can be displayed directly by simply printing the result object:

result

#%%
# We can see that our model does not perform well, as can be seen in the "Class Performance" check under the
# "Didn't Pass" section of the suite results. This is because the model was trained on a different dataset, and
# the model was not trained to detect tomatoes.
# Moreover, we can see that lowering the IoU threshold could have fixed this a bit (as can be seen in the
# "Mean Average Precision Report" Check), but would still keep the overall precision low.
# Moreover, under the "Passed" section, we can see that our drift checks have passed, which means that the distribution
# of the predictions on the training and test data is similar, and the issue is not there but in the model itself.