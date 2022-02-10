from abc import abstractmethod
from typing import Tuple

import PIL
from PIL import ImageChops
import numpy
import numpy as np
import plotly.graph_objects as go
import torch

from deepchecks.core.errors import DeepchecksValueError


# class ImageInfo:
#
#     @classmethod
#     @abstractmethod
#     def get_size(cls, img) -> Tuple[int, int]:
#         pass
#
#     @classmethod
#     @abstractmethod
#     def get_dimension(cls, img) -> int:
#         """Return the number of dimensions of the image (grayscale = 1, RGB = 3)."""
#         pass
#
#     @classmethod
#     @abstractmethod
#     def is_equals(cls, img) -> bool:
#         pass
#
#
# class ImageInfoNumpy(ImageInfo):
#
#     @classmethod
#     def get_dimension(cls) -> int:
#         pass
#
#     @classmethod
#     def is_equals(cls) -> bool:
#         pass
#
#     @classmethod
#     def get_size(cls) -> Tuple[int, int]:
#         pass


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


def get_image_dimension(img):
    if isinstance(img, (torch.Tensor, np.ndarray)):
        return img.shape[0]
    elif isinstance(img, PIL.Image.Image):
        return img.im.bands
    else:
        raise DeepchecksValueError(f'Don\'t know image object of type {type(img)}')


def get_image_size(img) -> Tuple[int, int]:
    """Return size of the image as (width, height)."""
    if isinstance(img, (torch.Tensor, np.ndarray)):
        return img.shape[1], img.shape[0]
    elif isinstance(img, PIL.Image.Image):
        return img.size
    else:
        raise DeepchecksValueError(f'Don\'t know image object of type {type(img)}')


def is_images_equal(img_a, img_b):
    """Test whether two given images are equal."""
    if isinstance(img_a, torch.Tensor):
        return torch.equal(img_a, img_b)
    elif isinstance(img_a, np.ndarray):
        return numpy.array_equal(img_a, img_b)
    elif isinstance(img_a, PIL.Image.Image):
        diff = ImageChops.difference(img_a, img_b)
        return diff.getbbox() is None
    else:
        raise DeepchecksValueError(f'Don\'t know image object of type {type(img_a)}')


def label_bbox_add_to_figure(label: torch.Tensor, figure, row=None, col=None, font_size=None):
    for single in label:
        clazz, x, y, w, h = single.tolist()
        # xs = [x, x, x + w, x + w, x]
        # ys = [y, y + h, y + h, y, y]
        figure.add_shape(type="rect", x0=x, y0=y, x1=x+w, y1=y+h, row=row, col=col, line=dict(color='red'))
        figure.add_annotation(x=x + w / 2, y=y, text=str(clazz), showarrow=False, yshift=10, row=row, col=col,
                              font=dict(color='red'))
