.. _supported_tasks_and_formats:

===========================
Supported Tasks and Formats
===========================

In order to accurately analyze the images, labels and predictions provided in the
:doc:`VisionData object </user-guide/vision/vision_data_class>`,
they must be provided in deepchecks accepted format. In this section we will describe the supported formats
for each supported task type.

The data formats are validated at the creation of the VisionData object based on the first batch of data. However,
in addition to the automatic validations, it is also important to make sure visually that the data was provided in the
correct format via the :func:`head <deepchecks.vision.VisionData.head>` functionality.


Common Formats
==============
All types of data are loaded in a batch by batch manner meaning that each data input contain
information about several samples. In order to support this, all data inputs are expected to be provided
as either an iterable in which each entry is the data for a single sample or as a high dimension array or tensor in
which the first dimension is the number of samples.

Image Format
------------
Each image in the batch can either be a `PIL image object <https://pillow.readthedocs.io/en/stable/reference/Image.html>`_
or a ``[H, W, C]`` 3D array (can be either numpy or a tensor).
For the array format, the first dimension must be the image y axis, the second being the image x axis, and the
third being the number of channels.
The numbers in the array should be integers in the range [0, 255]. Color images should be in RGB format and
have 3 channels, while grayscale images should have 1 channel.


Image Identifiers Format
------------------------
Image identifiers is a mechanism that allow identifying relevant images in the check results so that they can
be easily retrieved from the original dataset by the user. Common uses for this can be identifying images by
the path to the image or by their name or id in a database.
Image identifiers need to be provided as an iterable of strings.

.. _supported_tasks__classification:
Classification
==============
Label Format
------------
Classification label per sample is expected to be provided as either a string or an int representing the class index.

Prediction Format
-----------------
Classification prediction per sample should be **prediction probabilities** per class, meaning that it should be an
iterable of floats in the range [0, 1] with length equal to the number of classes.

.. _supported_tasks__object_detection:
Object Detection
================
Label Format
------------
Object detection label per sample should be an iterable of bounding boxes. Each bounding box should be an iterable
of 5 elements in the following order: ``(class_id, x_min, y_min, w, h)``.

    ``x_min`` and ``y_min`` are the coordinates (in pixels) of the **top left corner** of the bounding box, ``w``
    and ``h`` are the width and height of the bounding box (in pixels) and ``class_id`` is the class id of the label.

For example, labels of a batch containing 2 samples, where the first sample has a single bounding box and the second
sample has 2 bounding boxes, should be provided as follows:

.. code-block:: python

    [
        [(1, 50, 20, 9.4, 5.5)],
        [(3, 12, 23, 3.4, 9.5), (5, 42, 63, 9.4, 11.5)]
    ]

Prediction Format
-----------------
Object detection prediction per sample should be an iterable of predicted bounding boxes and their corresponding
confidence provided by the model. Each bounding box should be an iterable of 6 elements in the following order:
``(x_min, y_min, w, h, confidence, class_id)``.

    ``x_min``, ``y_min``, ``w`` and ``h`` represent the bounding box location as above,
    ``confidence`` is the confidence score given by
    the model to bounding box and ``class_id`` is the class id predicted by the model.

For example, predictions of a batch containing 2 samples, where the first sample has two predicted bounding boxes and
the second one has no predicted bounding boxes, should be provided as follows:

.. code-block:: python

    [
        [(50, 20, 9.4, 5.5, 0.8, 3), (50, 20, 9.4, 5.5, 0.4, 1)],
        []
    ]

.. _supported_tasks__segmentation:
Semantic Segmentation
=====================
Label Format
------------
Semantic segmentation label per sample should be an array of shape ``[H, W]`` where ``H`` and ``W`` are the
height and width of the corresponding image, and its values are the true class_ids of
the corresponding pixels in that image.
Note that the array should be 2D, as the number of channels on the original image are irrelevant to the label.

Prediction Format
-----------------
Semantic segmentation prediction per sample should be **predicted probabilities** per class per pixel. Specifically,
the prediction per sample format is an array of shape ``[C, H, W]`` where ``H`` and ``W`` are the height
and width of the corresponding image, and ``C`` is the number of possible classes
such that each channel corresponds to a class_id.

Note that the values of dimension C are the probabilities for each class and as such for each pixel the sum of the
probabilities should be 1.

Other Tasks
===========
For other tasks, the label and prediction formats are not validated nor used by the different checks.
However, there are many checks that require only the images themself. Few examples for such checks include:
:doc:`Image Property Outliers </checks_gallery/vision/data_integrity/plot_image_property_outliers.html>`,
:doc:`Image Dataset Drift </checks_gallery/vision/train_test_validation/plot_image_dataset_drift.html>` and
:doc:`Image Property Drift </checks_gallery/vision/train_test_validation/plot_image_property_drift.html>`.

In addition it is possible to generate custom metrics and properties for the predictions and labels
provided and run additional checks. For more information on how to do so, see the
:doc:`custom task tutorial </user-guide/vision/tutorials/custom_task_tutorial>`.
