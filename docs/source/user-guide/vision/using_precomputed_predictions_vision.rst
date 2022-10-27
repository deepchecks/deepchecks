.. _using_precomputed_predictions:

=================================
Using Pre-computed Predictions
=================================
Some checks, mainly the ones related to model evaluation, require model predictions in order to run.
In deepchecks, predictions are passed to the suite / check ``run`` method in one of the following ways:

* Implementing an ``infer_on_batch`` methods in the
  :doc:`VisionData object </user-guide/vision/data-classes/VisionData>`, or one of the child classes:
  :doc:`ClassificationData </user-guide/vision/data-classes/ClassificationData>`,
  :doc:`DetectionData </user-guide/vision/data-classes/DetectionData>`
  that allows the checks to compute the predictions.
* Passing the pre-computed predictions as a parameter to the check's ``run``.

Passing pre-computed predictions is a simple alternative to using a model in ``infer_on_batch``.
It is specifically recommended to use this option if your model object is unavailable locally (for example if placed on
a separate prediction server) or if the predicting process is computationally expensive or time consuming.

The pre-calculated predictions should be passed to suite/check's ``run`` method in the appropriate format.
The parameters to pass are ``predictions`` for single dataset checks and or ``train_predictions`` and
``test_predictions`` for checks that use both datasets.

Pre-computed Predictions Format
-------------------------------
The expected format is a dictionary of ``{sample index (int): sample predictions (tensor or list of tensors)}``
The accepted sample predictions format is according to the task:

* Classification: a tensor of shape (N_classes)
* Object Detection: a list of tensors, where each tensor is a bounding box in the format
  [x, y, w, h, confidence, class_id], where x and y are the coordinates of the top left corner, and x, y, w, h are in
  pixels.

Code Example
------------
In this example, we will compute and save the predictions on the MNIST dataset and then pass them to the
:doc:`ClassPerformance check</checks_gallery/vision/model_evaluation/plot_class_performance>` check as pre-computed
predictions.

Let's load the MNIST dataset and a pretrained classification model.

.. literalinclude:: ../../../../examples/vision/precomputed_predictions_examples.py
    :language: python
    :lines: 2-5
    :tab-width: 0

Now we will iterate over the datasets and save the predictions:

.. literalinclude:: ../../../../examples/vision/precomputed_predictions_examples.py
    :language: python
    :lines: 8-21
    :tab-width: 0

Next we will pass the saved predictions to the check and view the result:

.. literalinclude:: ../../../../examples/vision/precomputed_predictions_examples.py
    :language: python
    :lines: 24-26
    :tab-width: 0

Note that when passing the pre-computed predictions, you still need to pass the dataset(s) for additional data that the
check requires such as the labels.
