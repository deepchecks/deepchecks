.. _data_classes:

====================
Data Classes
====================

The different data classes in the deepchecks.vision module were designed to enable access to the data in a unified
format, while making the formatting code easy and maintainable. They are responsible for converting the data (whether
these are images, labels, or predictions) to a format deepchecks can work with.

.. toctree::
    :maxdepth: 2

    VisionData
    ClassificationData
    DetectionData
    SegmentationData

Why Is This Important?
========================

There is no standard when it comes to data formats in computer vision. Each model can have its own supported data format
for the output predictions, and each new benchmark dataset can propose a new labeling format. Even the images formats
can be different from each other (PIL, OpenCV, etc.).

In order to run a model and data-agnostic comprehensive testing suites, the data must be in an accepted format the
checks can work with. The data class objects provide a structured and repeatable way to do that, and are an important
part of the vision module.

What Do You Need to Implement?
==============================

Generally, all you need to do is to implement a data class that inherits from one of the supported classes in the
vision module, and just implement there 3 functions:

- ``batch_to_images``: Returns a list of images in the correct format from a batch of data.
- ``batch_to_labels``: Returns a list of labels in the correct format from a batch of data.
- ``infer_on_batch``: Returns a list of a model's predictions in the correct format from a batch of data.

Please see specific examples in the :doc:`VisionData </user-guide/vision/data-classes/VisionData>`,
:doc:`ClassificationData </user-guide/vision/data-classes/ClassificationData>`,
:doc:`DetectionData </user-guide/vision/data-classes/DetectionData>`,
and :doc:`SegmentationData </user-guide/vision/data-classes/SegmentationData>` classes.

Validating Your Data Class Implementation
=========================================
While implementing the data class, you may need a way to test your work is correct.
For this purpose, deepchecks contains an helper function which tests your implementation and prints its outputs,
which you can inspect to make sure everything works correctly.
Use this function like so:

.. code-block:: python

    from deepchecks.vision.utils.validation import validate_extractors
    dataset = MyDataset(my_dataloader)
    validate_extractors(dataset, model)

For more info about the classes validation see the following tutorial about how to
:doc:`/user-guide/vision/auto_tutorials/plot_extractors_validating`.