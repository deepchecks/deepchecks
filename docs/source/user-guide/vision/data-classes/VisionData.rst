.. _vision_data_class:

========================
The Vision Data Class
========================

The VisionData :doc:`data class </user-guide/vision/data-classes/index>` is the base class for all computer vision datasets, and represent a base CV task in deepchecks.
It wraps PyTorch DataLoader together with model related metadata and contains additional data and general methods
intended for easily accessing metadata relevant for validating a computer vision ML models.

For more info, please visit the API reference page: :class:`~deepchecks.vision.VisionData`

.. note::
    The VisionData class represents a base CV task, and ignores the labels of the dataset, and the predictions
    of the model. It is mainly used for checks that doesn't require them, such as
    :class:`~deepchecks.vision.checks.distribution.ImagePropertyDrift`

Class Properties
=================

The common properties are:

- **label_map** - A dictionary mapping class ids to their names.
- **name** - The dataset name to present in the displays instead of Train or Test.

Accepted Image Format
---------------------
All checks in deepchecks require images in the same format. They use the :func:`~deepchecks.vision.VisionData.batch_to_images` function in order to get
the images in the correct format.

The accepted format for a batch of images is an iterable of cv2 images. Each image in the iterable must be a [H, W, C] 3D numpy array.
The first dimension must be the image y axis, the second being the image x axis, and the third being
the number of channels. The numbers in the array should be in the range [0, 255]. Color images should be in RGB format
and have 3 channels, while grayscale images should have 1 channel. The dtype of the array should be uint8.

Examples
--------

.. code-block:: python

    from deepchecks.vision import VisionData

    class NormalizedImagesData(VisionData):
        """Implement a VisionData class for PIL images."""

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

    # Now, in order to test the class, we can create an instance of it:
    data = NormalizedImagesData(your_dataloader)

    # And validate the implementation:
    data.validate()

