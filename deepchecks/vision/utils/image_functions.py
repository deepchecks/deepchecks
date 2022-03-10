# ----------------------------------------------------------------------------
# Copyright (C) 2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module for defining functions related to image data."""
import base64
from typing import Tuple

import cv2
import numpy as np
import plotly.graph_objects as go
import torch

from deepchecks.core.errors import DeepchecksValueError


__all__ = ['ImageInfo', 'numpy_to_image_figure', 'label_bbox_add_to_figure', 'numpy_grayscale_to_heatmap_figure',
           'apply_heatmap_image_properties', 'numpy_to_html_image', 'crop_image']


class ImageInfo:
    """Class with methods defined to extract metadata about image."""

    def __init__(self, img):
        if not isinstance(img, np.ndarray):
            raise DeepchecksValueError('Expect image to be numpy array')
        self.img = img

    def get_size(self) -> Tuple[int, int]:
        """Get size of image as (width, height) tuple."""
        return self.img.shape[1], self.img.shape[0]

    def get_dimension(self) -> int:
        """Return the number of dimensions of the image (grayscale = 1, RGB = 3)."""
        return self.img.shape[2]

    def is_equals(self, img_b) -> bool:
        """Compare image to another image for equality."""
        return np.array_equal(self.img, img_b)


def numpy_to_image_figure(data: np.ndarray):
    """Create image graph object from given numpy array data."""
    dimension = data.shape[2]
    if dimension == 1:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    elif dimension != 3:
        raise DeepchecksValueError(f'Don\'t know to plot images with {dimension} dimensions')

    return go.Image(z=data, hoverinfo='skip')


def numpy_to_html_image(data: np.ndarray, labels=None):
    """Use plotly to create PNG image out of numpy data.

    Returns
    ------
    str
        HTML img tag with the embedded picture
    """
    dimension = data.shape[2]
    if dimension == 1:
        fig = go.Figure(go.Heatmap(z=data.squeeze(), colorscale='gray', hoverinfo='skip'))
        apply_heatmap_image_properties(fig)
        fig.update_traces(showscale=False)
    elif dimension == 3:
        fig = go.Figure(go.Image(z=data, hoverinfo='skip'))
    else:
        raise DeepchecksValueError(f'Don\'t know to plot images with {dimension} dimensions')

    if labels:
        label_bbox_add_to_figure(labels, fig)

    fig.update_yaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
    fig.update_xaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
    fig.update_layout(margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
    png = base64.b64encode(fig.to_image('png')).decode('ascii')
    style = 'position: absolute;margin: auto;top: 0;bottom: 0;left: 0; right: 0;max-width:100%;max-height:100%;'
    return f'<img style="{style}" src="data:image/png;base64, {png}">'


def numpy_grayscale_to_heatmap_figure(data: np.ndarray):
    """Create heatmap graph object from given numpy array data."""
    dimension = data.shape[2]
    if dimension == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    elif dimension != 1:
        raise DeepchecksValueError(f'Don\'t know to plot images with {dimension} dimensions')
    return go.Heatmap(z=data.squeeze(), hoverinfo='skip', coloraxis='coloraxis')


def apply_heatmap_image_properties(fig):
    """For heatmap and grayscale images, need to add those properties which on Image exists automatically."""
    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
    fig.update_xaxes(constrain='domain')


def label_bbox_add_to_figure(labels: torch.Tensor, figure, row=None, col=None, color='red',
                             prediction=False):
    """Add a bounding box label and rectangle to given figure."""
    for single in labels:
        if prediction:
            x, y, w, h, _, clazz = single.tolist()
        else:
            clazz, x, y, w, h = single.tolist()
        figure.add_shape(type='rect', x0=x, y0=y, x1=x+w, y1=y+h, row=row, col=col, line=dict(color=color))
        figure.add_annotation(x=x + w / 2, y=y, text=str(clazz), showarrow=False, yshift=10, row=row, col=col,
                              font=dict(color=color))


def crop_image(img: np.array, x, y, w, h) -> np.array:
    # Convert xywh to integers if not integers already:
    x, y, w, h = [round(n) for n in [x, y, w, h]]
    return img[y:y + h, x:x + w]
