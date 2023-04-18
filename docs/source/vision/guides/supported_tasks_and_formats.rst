.. _supported_tasks_and_formats:

===========================
Supported Tasks and Formats
===========================

In order to analyze the images, labels and predictions provided in the
:doc:`VisionData object </user-guide/vision/VisionData>`,
they must be provided in the accepted deepchecks format. In this section we will describe the supported formats
for each supported task type.

The VisionData object automatically validates your data formats when initialized, using the first batch of data. However,
in addition to the automatic validations, it is also important to make sure visually that the data was provided in the
correct format via the :func:`VisionData.head() <deepchecks.vision.vision_data.VisionData.head>` functionality.


.. attention:: Shuffling your Data
   In addition for the data being in the correct format, the data must also be loaded in a **shuffled** manner.
   When using PyTorch DataLoaders, deepchecks will by default attempt reshuffle the data. However, if you are using any
   other framework, you must make sure to shuffle the data yourself before passing it to deepchecks. The behaviour of
   reshuffling the data can be controlled via the
   ``reshuffle_data`` argument of the :class:`VisionData <deepchecks.vision.vision_data.VisionData>` constructor,
   and it should be manually set to `False` for non-PyTorch data.

We will first describe the formats which are common for all tasks, and then describe the label and
prediction formats which vary between tasks.

Common Formats
==============

Batch General Format
--------------------
Each batch, which is the output of the :doc:`batch loader </user-guide/vision/VisionData>`,
is required to be a :class:`dictionary <deepchecks.vision.vision_data.BatchOutputFormat>`
with keys for each provided data input: images, labels, predictions and image identifiers.
Since each data input represents multiple samples,
all data inputs are expected to be provided
as either an iterable in which each entry represents a single sample, or as a high dimension array or tensor in
which the first dimension is the number of samples.

That means that the following are all valid batch formats for a single batch:

.. code-block:: python

    {
        'images': [image1, image2, ...],
        'labels': [label1, label2, ...],
        'predictions': [prediction1, prediction2, ...],
        'image_identifiers': [image_identifier1, image_identifier2, ...]
    }

Or:

.. code-block:: python

    from deepchecks.vision.vision_data import BatchOutputFormat

    BatchOutputFormat(
        images=[image1, image2, ...],
        labels=[label1, label2, ...],
        predictions=[prediction1, prediction2, ...],
        image_identifiers=[image_identifier1, image_identifier2, ...]
    )

Image Format
------------
Each image in the batch can either be a
`PIL image object <https://pillow.readthedocs.io/en/stable/reference/Image.html>`_
or a ``[H, W, C]`` 3D array (can be either numpy or a tensor),
where ``H``, ``W`` and ``C`` represent the height, width and channel of the image.
The array values should be integers in the range [0, 255]. Color images should be in RGB format and
have 3 channels, while grayscale images should have 1 channel.


Image Identifiers Format
------------------------
Image identifiers is a mechanism that allow identifying relevant images in the check results so that they can
be easily retrieved later on from the original dataset by the user.

Common implementations for this can be the path to the image or the image name or id in a database.
Image identifiers need to be provided as an iterable of strings.

.. code-block:: python

    ['s3://my_images/dogs/1.jpg', 's3://my_images/cats/2.jpg', ...]

.. _supported_tasks__classification:
Classification
==============

Label Format
------------
Classification label per sample is expected to be provided as either a string or an int
representing the class index. For example, labels for a batch containing 2 samples can be provided as:

.. code-block:: python

    [1, 3]

Prediction Format
-----------------
Classification prediction for each sample should be the **prediction probabilities** per class, meaning that it should be an
iterable of floats in the range [0, 1] with length equal to the number of classes.

For example, predictions of a batch containing 2 samples for a classification task with 3 possible
classes may look like this:

.. code-block:: python

    [
        [0.1, 0.2, 0.7],
        [0.3, 0.6, 0.1]
    ]


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
        [(1, 50, 20, 9, 5)],
        [(3, 12, 23, 3, 9), (5, 42, 63, 9, 11)]
    ]

Prediction Format
-----------------
Object detection prediction per sample should be an iterable of predicted bounding boxes and their corresponding
confidence scores provided by the model. Each bounding box should be an iterable of 6 elements in the following order:
``(x_min, y_min, w, h, confidence, class_id)``.

    ``x_min``, ``y_min``, ``w`` and ``h`` represent the bounding box location as above,
    ``confidence`` is the confidence score given by
    the model to the bounding box and ``class_id`` is the class id predicted by the model.

For example, predictions of a batch containing 2 samples, where the first sample has two predicted bounding boxes and
the second one has no predicted bounding boxes, should be provided as follows:

.. code-block:: python

    [
        [(33, 11, 7, 9, 0.8, 3), (50, 20, 9, 5, 0.4, 1)],
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
For other tasks, there is no specific format required for the labels and predictions and their format is
not validated. There are two ways in which Deepchecks can provide value for these sort of tasks:

The quick option: Run checks that require only the images themselves.
Few examples for such checks include:
:doc:`Image Property Outliers </checks_gallery/vision/data_integrity/plot_image_property_outliers>`,
:doc:`Image Dataset Drift </checks_gallery/vision/train_test_validation/plot_image_dataset_drift>` and
:doc:`Image Property Drift </checks_gallery/vision/train_test_validation/plot_image_property_drift>`.

The advanced option: Add custom metrics and properties for the
predictions and labels provided and run additional checks.
For more information on how to do so, see the
:doc:`custom task tutorial </user-guide/vision/auto_tutorials/plot_custom_task_tutorial>`.
