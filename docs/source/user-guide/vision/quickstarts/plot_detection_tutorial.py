"""
.. _vision_detection_tutorial:

==========================
Object Detection Tutorial
==========================

In this tutorial, you will learn how to validate your **object detection model** using deepchecks test suites.
You can read more about the different checks and suites for computer vision use cases at the
:doc:`examples section  </checks_gallery/vision/index>`

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

# Importing the required packages
import os
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from functools import partial

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

from deepchecks.vision.detection_data import DetectionData

#%%
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
        bboxes = []
        labels = []
        with open(ann_path, 'r') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

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
train_set, test_set = torch.utils.data.random_split(dataset,
                                                    [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)],
                                                    generator=torch.Generator().manual_seed(42))
test_set.transforms = A.Compose([ToTensorV2()])
train_loader = DataLoader(train_set, batch_size=64, collate_fn=(lambda batch: tuple(zip(*batch))))
test_loader = DataLoader(test_set, batch_size=64, collate_fn=(lambda batch: tuple(zip(*batch))))

#%%
# Visualize a Few Images
# ~~~~~~~~~~~~~~~~~~~~~~
# Let's visualize a few training images so as to understand the data augmentation.

def prepare(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1) * 255
    inp = inp.transpose((2,0,1))
    return torch.tensor(inp, dtype=torch.uint8)

import torchvision.transforms.functional as F


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(20,20))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

from torchvision.utils import draw_bounding_boxes

data = next(iter(train_loader))
inp, targets = data[0][:4], data[1][:4]


result = [draw_bounding_boxes(prepare(inp[i]), torch.stack(targets[i]['boxes']),
                              colors=['yellow'] * torch.stack(targets[i]['boxes']).shape[0], width=5)
          for i in range(len(targets))]
show(result)

#%%
# .. image :: /_static/images/tutorials/tomatoes.png
#     :alt: Tomatoes with bbox
#
# Downloading a Pre-trained Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this tutorial, we will download a pre-trained SSDlite model and a MobileNetV3 Large backbone
# from the official PyTorch repository. For more details, please refer to the
# `official documentation <https://pytorch.org/vision/stable/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html#torchvision.models.detection.ssdlite320_mobilenet_v3_large>`_.
#
# After downloading the model, we will fine-tune it for our particular classes. We will do it by replacing the pre-trained
# head with a new one that matches our needs.

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)

in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
num_anchors = model.anchor_generator.num_anchors_per_location()
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
model.to(device)

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
# Visualize the Data Loader and the Model Outputs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# First we'll make sure we are familiar with the data loader and the model outputs.

batch = next(iter(train_loader))

print("Batch type is: ", type(batch))
print("First element is: ", type(batch[0]), "with len of ", len(batch[0]))
print("Example output of an image shape from the dataloader ", batch[0][0].shape)
print("Image values", batch[0][0])
print("-"*80)

print("Second element is: ", type(batch[1]), "with len of ", len(batch[1]))
print("Example output of a label from the dataloader ", batch[1][0])

#%%
# Implementing the DetectionData class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The checks in the package validate the model & data by calculating various quantities over the data, labels and
# predictions. In order to do that, those must be in a pre-defined format, according to the task type.
# The first step is to implement a class that enables deepchecks to interact with your model and data and transform
# them to this pre-defined format, which is set for each task type.
# In this tutorial, we will implement the object detection task type by implementing a class that inherits from the
# :class:`deepchecks.vision.detection_data.DetectionData` class.
#
# The DetectionData class contains additional data and general methods intended for easy access to relevant metadata
# for object detection ML models validation.
# To learn more about the expected format please visit the API reference for the
# :class:`deepchecks.vision.detection_data.DetectionData` class.

from deepchecks.vision.detection_data import DetectionData


class TomatoData(DetectionData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_to_images(self, batch):
        """
        Convert a batch of data to images in the expected format. The expected format is an iterable of cv2 images,
        where each image is a numpy array of shape (height, width, channels). The numbers in the array should be in the
        range [0, 255] in a uint8 format.
        """
        inp = torch.stack(list(batch[0])).cpu().detach().numpy().transpose((0, 2, 3, 1))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Un-normalize the images
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp * 255

    def batch_to_labels(self, batch):
        """
        Convert a batch of data to labels in the expected format. The expected format is a list of tensors of length N,
        where N is the number of samples. Each tensor element is in a shape of [B, 5], where B is the number of bboxes
        in the image, and each bounding box is in the structure of [class_id, x, y, w, h].
        """
        tensor_annotations = batch[1]
        label = []
        for annotation in tensor_annotations:
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

    def infer_on_batch(self, batch, model, device):
        """
        Returns the predictions for a batch of data. The expected format is a list of tensors of shape length N, where N
        is the number of samples. Each tensor element is in a shape of [B, 6], where B is the number of bboxes in the
        predictions, and each bounding box is in the structure of [x, y, w, h, score, class_id].
        """
        nm_thrs = 0.2
        score_thrs = 0.7
        imgs = list(img.to(device) for img in batch[0])
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
# After defining the task class, we can validate it by running the following code:

# We have a single label here, which is the tomato class
# The label_map is a dictionary that maps the class id to the class name, for display purposes.
LABEL_MAP = {
    1: 'Tomato'
}
training_data = TomatoData(data_loader=train_loader, label_map=LABEL_MAP)
test_data = TomatoData(data_loader=test_loader, label_map=LABEL_MAP)

training_data.validate_format(model, device=device)
test_data.validate_format(model, device=device)

# And observe the output:

#%%
# Running Deepchecks' suite on our data and model!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now that we have defined the task class, we can validate the model with the deepchecks' model evaluation suite.
# This can be done with this simple few lines of code:

from deepchecks.vision.suites import model_evaluation

suite = model_evaluation()
result = suite.run(training_data, test_data, model, device=device)

#%%
# We also have suites for:
# :func:`data integrity <deepchecks.vision.suites.default_suites.data_integrity>`
# - validating a single dataset and
# :func:`train test validation <deepchecks.vision.suites.default_suites.train_test_validation>` -
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
