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
import io
import typing as t

import cv2
import numpy as np
import PIL.Image as pilimage
import PIL.ImageDraw as pildraw
import PIL.ImageOps as pilops
import plotly.graph_objects as go

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.html import imagetag
from deepchecks.vision.vision_data import TaskType

from .detection_formatters import convert_bbox

__all__ = ['numpy_grayscale_to_heatmap_figure', 'ensure_image',
           'apply_heatmap_image_properties', 'draw_bboxes', 'prepare_thumbnail',
           'crop_image', 'draw_image']


def draw_image(image: np.ndarray, label, task_type: TaskType,
               thumbnail_size: t.Tuple[int, int] = (200, 200), draw_label: bool = True) -> str:
    """Return an image to show as output of the display.

    Parameters
    ----------
    image : np.ndarray
        The image to draw, must be a [H, W, C] 3D numpy array.
    label :
        2-dim labels tensor for the image to draw on top of the image, shape depends on task type.
    task_type : TaskType
        The task type associated with the label.
    thumbnail_size: t.Tuple[int,int]
        The required size of the image for display.
    draw_label : bool, default: True
        Whether to draw the label on the image or not.
    Returns
    -------
    str
        The image in the provided thumbnail size with the label drawn on top of it for relevant tasks as html.
    """
    if label is not None and image is not None and draw_label and task_type == TaskType.OBJECT_DETECTION:
        image = draw_bboxes(image, np.asarray(label), copy_image=False, border_width=5)
    if image is not None:
        return prepare_thumbnail(image=image, size=thumbnail_size, copy_image=False)
    else:
        return 'Image unavailable'


def ensure_image(
        image: t.Union[pilimage.Image, np.ndarray],
        copy: bool = True
) -> pilimage.Image:
    """Transform to `PIL.Image.Image` if possible.

    Parameters
    ----------
    image : Union[PIL.Image.Image, numpy.ndarray]
    copy : bool, default True
        if `image` is an instance of the `PIL.Image.Image` return
        it as it is or copy it.

    Returns
    -------
    `PIL.Image.Image`
    """
    if isinstance(image, pilimage.Image):
        return image.copy() if copy is True else image
    if isinstance(image, np.ndarray):
        image = image.squeeze().astype(np.uint8)
        if image.ndim == 3:
            return pilimage.fromarray(image)
        elif image.ndim == 2:
            return pilops.colorize(
                pilimage.fromarray(image),
                black='black',
                white='white',
                blackpoint=image.min(),
                whitepoint=image.max(),
            )
        else:
            raise ValueError(f'Do not know how to work with {image.ndim} dimensional images')
    else:
        raise TypeError(f'cannot convert {type(image)} to the PIL.Image.Image')


def draw_bboxes(
        image: t.Union[pilimage.Image, np.ndarray],
        bboxes: np.ndarray,
        bbox_notation: t.Optional[str] = None,
        copy_image: bool = True,
        border_width: int = 1,
        color: t.Union[str, t.Dict[np.number, str]] = 'red',
) -> pilimage.Image:
    """Draw bboxes on the image.

    Parameters
    ----------
    image : Union[PIL.Image.Image, numpy.ndarray]
        image to draw on
    bboxes : numpy.ndarray
        array of bboxes
    bbox_notation : Optional[str], default None
        format of the provided bboxes
    copy_image : bool, default True
        copy image before drawing or not
    border_width : int, default 1
        width of the bbox outline
    color: Union[str, Dict[number, str]], default "red"
        color of the bbox outline. It could be a map mapping class id to the color

    Returns
    -------
    PIL.Image.Image : image instance with drawen bboxes on it
    """
    image = ensure_image(image, copy=copy_image)
    draw = pildraw.ImageDraw(image)

    if len(bboxes.shape) == 1:
        bboxes = [bboxes]
    if bbox_notation is not None:
        bboxes = np.array(
            [convert_bbox(bbox, notation=bbox_notation, image_width=image.width, image_height=image.height,
                          _strict=False).tolist() for bbox in bboxes])
    for bbox in bboxes:
        clazz, x0, y0, w, h = bbox
        x1, y1 = x0 + w, y0 + h

        if isinstance(color, str):
            color_to_use = color
        elif isinstance(color, dict):
            color_to_use = color[clazz]
        else:
            raise TypeError('color must be of type - Union[str, Dict[int, str]]')

        draw.rectangle(xy=(x0, y0, x1, y1), width=border_width, outline=color_to_use)
        draw.text(xy=(x0 + (w * 0.5), y0 + (h * 0.2)), text=str(clazz), fill=color_to_use)

    return image


def prepare_thumbnail(
        image: t.Union[pilimage.Image, np.ndarray],
        size: t.Optional[t.Tuple[int, int]] = None,
        copy_image: bool = True,
) -> str:
    """Prepare html image tag with the provided image.

    Parameters
    ----------
    image : Union[PIL.Image.Image, numpy.ndarray]
        image to use
    size : Optional[Tuple[int, int]], default None
        size to which image should be rescaled
    copy_image : bool, default True
        to rescale the image to the provided size this function uses
        `PIL.Image.Image.thumbnail` method that modified image instance
        in-place. If `copy_image` is set to True image will be copied
        before rescaling.

    Returns
    -------
    str : html '<img>' tag with embedded image
    """
    if size is not None:
        image = ensure_image(image, copy=copy_image)
        # First define the correct size with respect to the original aspect ratio
        width_factor = size[0] / image.size[0]
        height_factor = size[1] / image.size[1]
        # Takes the minimum factor in order for the image to not exceed the size in either width or height
        factor = min(width_factor, height_factor)
        size = (int(image.size[0] * factor), int(image.size[1] * factor))
        # Resize the image
        image = image.resize(size, pilimage.ANTIALIAS)
    else:
        image = ensure_image(image, copy=False)

    img_bytes = io.BytesIO()
    image.save(img_bytes, optimize=True, quality=60, format='jpeg')
    img_bytes.seek(0)
    tag = imagetag(img_bytes.read())
    img_bytes.close()
    return tag


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
    fig.update_yaxes(autorange='reversed', constrain='domain')
    fig.update_xaxes(constrain='domain')


def crop_image(img: np.ndarray, x, y, w, h) -> np.ndarray:
    """Return the cropped numpy array image by x, y, w, h coordinates (top left corner, width and height."""
    # Convert x, y, w, h to integers if not integers already:
    x, y, w, h = [round(n) for n in [x, y, w, h]]

    # Make sure w, h don't extend the bounding box outside of image dimensions:
    h = min(h, img.shape[0] - y - 1)
    w = min(w, img.shape[1] - x - 1)

    return img[y:y + h, x:x + w]
