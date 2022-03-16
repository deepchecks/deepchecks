.. _detection_data_class:

===============================
The Object Detection Data Class
===============================

The DetectionData class represents a CV object detection task in deepchecks.
It is a subclass of the :class:`~deepchecks.vision.VisionData` class and is used to load and preprocess data for an
object detection task.
The DetectionData class is containing additional data and general methods intended for easily accessing metadata
relevant for validating a computer vision object detection ML models.

For more info, please visit the API reference page: :class:`~deepchecks.vision.DetectionData`

Accepted Image Format
---------------------
All checks in deepchecks require images in the same format. They use the `batch_to_images` function in order to get
the images in the correct format. For more info on the accepted formats, please visit the
`VisionData User Guide <VisionData.rst>`_.

Accepted Label Format
---------------------
Deepchecks' checks use the `batch_to_labels` function in order to get the labels in the correct format.
The accepted label format for is a a list of length N containing tensors of shape (B, 5), where N is the number
of samples, B is the number of bounding boxes in the sample and each bounding box is represented by 5 values:
``(class_id, x, y, w, h)``.
    x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w
    and h are the width and height of the bounding box (in pixels) and class_id is the class id of the prediction.

For example, for a sample with 2 bounding boxes, the label format may be:
``[(1, 8.4, 50.2, 100, 100), (5, 26.4, 10.1, 20, 40)]``.

Accepted Prediction Format
--------------------------
Deepchecks' checks use the `infer_on_batch` function in order to get the predictions of the model in the correct format.
The accepted prediction format is a list of length N containing tensors of shape (B, 6), where N is the number
of images, B is the number of bounding boxes detected in the sample and each bounding box is represented by 6
values: ``[x, y, w, h, confidence, class_id]``.
    x and y are the coordinates (in pixels) of the upper left corner
    of the bounding box, w and h are the width and height of the bounding box (in pixels), confidence is the
    confidence of the model and class_id is the class id.

For example, for a sample with 2 bounding boxes, the prediction format may be:
``[(8.4, 50.2, 100, 100, 0.9, 1), (26.4, 10.1, 20, 40, 0.8, 5)]``.

Examples
--------

.. code-block:: python

    from deepchecks.vision import DetectionData
    import torch.nn.functional as F
    import torch
    import numpy as np

    class MyDetectionTaskData(DetectionData)
    """Implement a ClassificationData class for a classification task."""

        def batch_to_images(self, batch):
            """Convert a batch of images to a list of PIL images.

            Parameters
            ----------
            batch : torch.Tensor
                The batch of images to convert.

            Returns
            -------
            list
                A list of PIL images.
            """

            # Assuming batch[0] is a batch of (N, C, H, W) images, we convert it to (N, H, W, C)/
            imgs = batch[0].detach().numpy().transpose((0, 2, 3, 1))

            # The images are normalized to [0, 1] range based on the mean and std of the ImageNet dataset, so we need to
            # convert them back to [0, 255] range.
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            imgs = std * imgs + mean
            imgs = np.clip(imgs, 0, 1)
            imgs *= 255

            return imgs

        def batch_to_labels(self, batch):
            """Convert a batch of labels to a tensor.

            Parameters
            ----------
            batch : torch.Tensor
                The batch of labels to convert.

            Returns
            -------
            torch.Tensor
                A tensor of shape (N,).
            """

            # each bbox in the labels is (class_id, x, y, x, y). convert to (class_id, x, y, w, h)
            return [torch.stack(
                   [torch.cat((bbox[0], bbox[1:3], bbox[4:] - bbox[1:3]), dim=0)
                       for bbox in image])
                    for image in batch[1]]

        def infer_on_batch(self, batch, model, device):
            """Get the predictions of the model on a batch of images.

            Parameters
            ----------
            batch : torch.Tensor
                The batch of data.
            model : torch.nn.Module
                The model to use for inference.
            device : torch.device
                The device to use for inference.

            Returns
            -------
            torch.Tensor
                A tensor of shape (N, n_classes).
            """

            # Converts a yolo prediction batch to the accepted xywh format
            return_list = []

            predictions = model(batch[0])
            # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
            for single_image_tensor in predictions.pred:
                pred_modified = torch.clone(single_image_tensor)
                pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]
                pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]
                return_list.append(pred_modified)

            return return_list

    # Now, in order to test the class, we can create an instance of it:
    data = MyDetectionTaskData(your_dataloader)

    # And validate the implementation:
    data.validate()

