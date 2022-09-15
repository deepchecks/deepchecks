.. _detection_data_class:

===============================
The Object Detection Data Class
===============================

The DetectionData is a :doc:`data class </user-guide/vision/data-classes/index>` designed for object detection tasks.
It is a subclass of the :class:`~deepchecks.vision.VisionData` class and is used to help deepchecks load and interact with object detection data using a well defined format.
detection related checks.

For more info, please visit the API reference page: :class:`~deepchecks.vision.DetectionData`

Accepted Image Format
---------------------

All checks in deepchecks require images in the same format. They use the :func:`~deepchecks.vision.VisionData.batch_to_images` function in order to get
the images in the correct format. For more info on the accepted formats, please visit the
:doc:`VisionData User Guide </user-guide/vision/data-classes/VisionData>`.

Accepted Label Format
---------------------

Deepchecks' checks use the :func:`~deepchecks.vision.DetectionData.batch_to_labels` function in order to get the labels in the correct format.
The accepted label format is a a list of length N containing tensors of shape (B, 5), where N is the number
of samples within a batch, B is the number of bounding boxes in the sample and each bounding box is represented by 5 values:
``(class_id, x_min, y_min, w, h)``.

    x_min and y_min are the coordinates (in pixels) of the **top left corner** of the bounding box, w
    and h are the width and height of the bounding box (in pixels) and class_id is the class id of the prediction.

For example, for a sample with 2 bounding boxes, the label format may be:
``tensor([[1, 8.4, 50.2, 100, 100], [5, 26.4, 10.1, 20, 40]])``.

Accepted Prediction Format
--------------------------

Deepchecks' checks use the :func:`~deepchecks.vision.DetectionData.infer_on_batch` function in order to get the predictions of the model in the correct format.
The accepted prediction format is a list of length N containing tensors of shape (B, 6), where N is the number
of images, B is the number of bounding boxes detected in the sample and each bounding box is represented by 6
values: ``[x_min, y_min, w, h, confidence, class_id]``.

    x_min,y_min,w and h represent the bounding box location as above, confidence is the confidence score given by the model to
    bounding box and class_id is the class id predicted by the model.

For example, for a sample with 2 bounding boxes, the prediction format may be:
``tensor([[8.4, 50.2, 100, 100, 0.9, 1], [26.4, 10.1, 20, 40, 0.8, 5]])``.

Example
--------

Assuming we have implemented a torch DataLoader whose underlying __getitem__ method returns a tuple of the form:
``(images, bboxes)``. ``images`` is a tensor of shape (N, C, H, W) in which the images pixel values are normalized to
[0, 1] range based on the mean and std of the ImageNet dataset. ``bboxes`` is a tensor of shape (N, B, 5) in which
each box arrives in the format: ``(class_id, x_min, y_min, x_max, y_max)``. Additionally, we are using Yolo as a model.

.. code-block:: python

    from deepchecks.vision import DetectionData
    import torch.nn.functional as F
    import torch
    import numpy as np

    class MyDetectionTaskData(DetectionData)
    """A deepchecks data digestion class for object detection related checks."""

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
            """Convert a batch bounding boxes to the required format.

            Parameters
            ----------
            batch : tuple
                The batch of data, containing images and bounding boxes.

            Returns
            -------
            List
                A list of size N containing tensors of shape (B,5).
            """

            # each bbox in the labels is (class_id, x, y, x, y). convert to (class_id, x, y, w, h)
            bboxes = []
            for bboxes_single_image in batch[1]:
                formatted_bboxes = [torch.cat((bbox[0], bbox[1:3], bbox[4:] - bbox[1:3]), dim=0)
                                    for bbox in bboxes_single_image]
                if len(formatted_bboxes) != 0:
                    bboxes.append(torch.stack(formatted_bboxes))
            return bboxes

        def infer_on_batch(self, batch, model, device):
            """Get the predictions of the model on a batch of images.

            Parameters
            ----------
            batch : tuple
                The batch of data, containing images and bounding boxes.
            model : torch.nn.Module
                The model to use for inference.
            device : torch.device
                The device to use for inference.

            Returns
            -------
            List
                A list of size N containing tensors of shape (B,6).
            """

            return_list = []
            predictions = model.to(device)(batch[0])

            # yolo Detections objects have List[torch.Tensor(B,6)] output where each bbox is
            #(x_min, y_min, x_max, y_max, confidence, class_id).
            for single_image_tensor in predictions.pred:
                pred_modified = torch.clone(single_image_tensor)
                pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]
                pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]
                return_list.append(pred_modified)

            return return_list

    # Now, in order to test the class, we can create an instance of it:
    data = MyDetectionTaskData(your_dataloader)

    # And validate the implementation:
    data.validate_format(your_model)

