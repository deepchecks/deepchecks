HuggingFace Transformers
========================

This tutorial demonstrates how deepchecks.vision can be used on a Hugging Face transformer model. We will use deepchecks
to compare the performance of the `DETR ResNet <https://huggingface.co/facebook/detr-resnet-50>`__ transformers model
against the widely used `YOLOv5s <https://arxiv.org/abs/1804.02767>`__ model on the `COCO <https://cocodataset.org/>`__
dataset.

Building a VisionData Object for the DETR Model
-----------------------------------------------

In order to use the DETR model, we need to translate the image, label and prediction formats to the ones supported by
deepchecks (see the :doc:`Format Guide </user-guide/vision/supported_tasks_and_formats>`) and define a
:class:`deepchecks.vision.vision_data.VisionData` object that will be used to run the checks.

We'll start by loading the DETR ResNet model from the Hugging Face Transformers library:

.. literalinclude:: ../../../../examples/integrations/hugging_face/deepchecks_hugging_face_tutorial.py
    :language: python
    :start-after: # LOAD DETR
    :end-before: # IMPLEMENT DETR INTEGRATION
    :tab-width: 0

And then we'll move on to implementing the COCODETRData class, which will help us keep all the required format
conversions in one place.

.. literalinclude:: ../../../../examples/integrations/hugging_face/deepchecks_hugging_face_tutorial.py
    :language: python
    :start-after: # IMPLEMENT DETR INTEGRATION
    :end-before: # CREATE VALIDATE DETR
    :tab-width: 0

We can now create :class:`VisionData <deepchecks.vision.vision_data.VisionData>` object. This deepchecks object accepts
a batch loader, which is an iterator that yields batches of images, labels, predictions and any other required
information. To read more about it, see the :doc:`Vision Data Guide </user-guide/vision/VisionData>`. In this example
our batch loader is a python dataloader, so we'll create a custom collate function that will convert the data to the
required formats and generate the predictions. We'll then use the :meth:`head <deepchecks.vision.vision_data.VisionData.head>`
method to make sure the dataset was created successfully.

.. literalinclude:: ../../../../examples/integrations/hugging_face/deepchecks_hugging_face_tutorial.py
    :language: python
    :start-after: # CREATE VALIDATE DETR
    :end-before: # LOAD YOLO
    :tab-width: 0

.. image:: /_static/detr_valid.png
   :alt: Validating
   :align: left

Great! We can see that the labels match the object locations, and that the labels an detections align.


Load Pre-Made YOLOv5s
---------------------

Next, we'll load from :mod:`deepchecks.vision.datasets.detection.coco_torch` a VisionData containing a sample of the COCO dataset (coco 128)
complete with YOLO predictions on this dataset, both downloaded from `ultralytics <https://github.com/ultralytics/yolov5>`__ repository. We'll use yolo
to benchmark the results achieved by the DETR model.

.. literalinclude:: ../../../../examples/integrations/hugging_face/deepchecks_hugging_face_tutorial.py
    :language: python
    :start-after: # LOAD YOLO
    :end-before: # CHECK ON YOLO
    :tab-width: 0

We already loaded the data wrapped with the relevant ``VisionData`` object, so we can just use the
:doc:`MeanAveragePrecisionReport </checks_gallery/vision/model_evaluation/plot_mean_average_precision_report>` check to
evaluate the model's performance for various IoU thresholds and bounding box sizes.

.. literalinclude:: ../../../../examples/integrations/hugging_face/deepchecks_hugging_face_tutorial.py
    :language: python
    :start-after: # CHECK ON YOLO
    :end-before: # CHECK ON DETR
    :tab-width: 0

.. image:: /_static/yolo_map.png
   :alt: Mean Average Precision Report for Yolov5
   :align: center



Benchmarking YOLOv5s Against DETR ResNet
------------------------------------------

Now that we have everything in place, we can run the
:doc:`MeanAveragePrecisionReport </checks_gallery/vision/model_evaluation/plot_mean_average_precision_report>` check
also on the DETR model! Let's run and compare to the YOLO results.

.. literalinclude:: ../../../../examples/integrations/hugging_face/deepchecks_hugging_face_tutorial.py
    :language: python
    :start-after: # CHECK ON DETR
    :end-before: # SHOW ON YOLO
    :tab-width: 0

.. image:: /_static/detr_map.png
   :alt: Mean Average Precision Report for DETR ResNet
   :align: center

Comparing to the results achieved earlier with YOLO:

.. literalinclude:: ../../../../examples/integrations/hugging_face/deepchecks_hugging_face_tutorial.py
    :language: python
    :start-after: # SHOW ON YOLO
    :end-before: # END
    :tab-width: 0

.. image:: /_static/yolo_map.png
   :alt: Mean Average Precision Report for Yolov5
   :align: center

We can clearly see an improvement in the DETR model! We can further see that the greatest improvement has been achieved
for the larger objects, with objects of sizes of up to 32^2 squared pixels improving only from an mAP of
0.21 to 0.26.

Of course, now that the VisionData object has been implemented you can use any of the other deepchecks check and suites.
You can check them out in our :doc:`check gallery </checks_gallery/vision>`, and learn more about
:doc:`when you should use </getting-started/when_should_you_use>` each of our built-in suites.
