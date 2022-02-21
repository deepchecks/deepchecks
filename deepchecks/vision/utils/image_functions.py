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
from typing import Tuple

import cv2
import numpy as np
import plotly.graph_objects as go
import torch

from deepchecks.core.errors import DeepchecksValueError


__all__ = ['ImageInfo', 'numpy_to_image_figure', 'label_bbox_add_to_figure']


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


def label_bbox_add_to_figure(label: torch.Tensor, figure, row=None, col=None, color='red',
                             prediction=False):
    """Add a bounding box label and rectangle to given figure."""
    for single in label:
        if prediction:
            x, y, w, h, _, clazz = single.tolist()
        else:
            clazz, x, y, w, h = single.tolist()
        figure.add_shape(type='rect', x0=x, y0=y, x1=x+w, y1=y+h, row=row, col=col, line=dict(color=color))
        figure.add_annotation(x=x + w / 2, y=y, text=str(clazz), showarrow=False, yshift=10, row=row, col=col,
                              font=dict(color=color))


def label_bbox_full_add_to_figure(label: torch.Tensor, figure, row=None, col=None, opacity=0.3):
    """Add full and transparent bounding box label and rectangle to given figure."""
    for single in label:
        _, x, y, w, h = single.tolist()
        figure.add_shape(type='rect', x0=x, y0=y, x1=x+w, y1=y+h, row=row, col=col, fillcolor='blue', opacity=opacity,
                         line_width=0)
