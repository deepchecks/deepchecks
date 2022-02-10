from abc import abstractmethod
from typing import Tuple

import PIL
from PIL import ImageChops
import numpy
import numpy as np
import plotly.graph_objects as go
import torch

from deepchecks.core.errors import DeepchecksValueError


__all__ = ['get_image_info', 'numpy_to_image_figure', 'apply_heatmap_image_properties', 'label_bbox_add_to_figure']


class ImageInfo:

    def __init__(self, img):
        self.img = img

    @abstractmethod
    def get_size(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the number of dimensions of the image (grayscale = 1, RGB = 3)."""
        pass

    @abstractmethod
    def is_equals(self, img_b) -> bool:
        pass


class ImageInfoNumpy(ImageInfo):

    def get_dimension(self) -> int:
        return self.img.shape[2]

    def is_equals(self, img_b) -> bool:
        return numpy.array_equal(self.img, img_b)

    def get_size(self) -> Tuple[int, int]:
        return self.img.shape[1], self.img.shape[0]


class ImageInfoPIL(ImageInfo):

    def get_dimension(self) -> int:
        return self.img.im.bands

    def is_equals(self, img_b) -> bool:
        diff = ImageChops.difference(self.img, img_b)
        return diff.getbbox() is None

    def get_size(self) -> Tuple[int, int]:
        return self.img.size


def numpy_to_image_figure(data: np.ndarray):
    # Image knows to plot only RGB images
    dimension = data.shape[2]
    if dimension == 3:
        return go.Image(z=data, hoverinfo='skip')
    elif dimension == 1:
        return go.Heatmap(z=data.squeeze(), colorscale='gray', hoverinfo='skip')
    else:
        raise DeepchecksValueError(f'Don\'t know to plot images with {dimension} dimensions')


def apply_heatmap_image_properties(fig):
    # In case of heatmap and grayscale images, need to add those properties which on Image exists automatically
    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
    fig.update_xaxes(constrain='domain')
    fig.update_traces(showscale=False)


def get_image_info(img):
    """Return object containing info about the image."""
    if isinstance(img, torch.Tensor):
        return ImageInfoNumpy(img.numpy())
    elif isinstance(img, np.ndarray):
        return ImageInfoNumpy(img)
    elif isinstance(img, PIL.Image.Image):
        return ImageInfoPIL(img)
    else:
        raise DeepchecksValueError(f'Don\'t know image object of type {type(img)}')


def label_bbox_add_to_figure(label: torch.Tensor, figure, row=None, col=None):
    for single in label:
        clazz, x, y, w, h = single.tolist()
        # xs = [x, x, x + w, x + w, x]
        # ys = [y, y + h, y + h, y, y]
        figure.add_shape(type="rect", x0=x, y0=y, x1=x+w, y1=y+h, row=row, col=col, line=dict(color='red'))
        figure.add_annotation(x=x + w / 2, y=y, text=str(clazz), showarrow=False, yshift=10, row=row, col=col,
                              font=dict(color='red'))
