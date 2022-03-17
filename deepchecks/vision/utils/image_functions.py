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
import typing as t
import io
import base64
from textwrap import dedent

import cv2
import numpy as np
import torch
import PIL.Image as pilimage
import PIL.ImageDraw as pildraw
import PIL.ImageOps as pilops
import plotly.graph_objects as go

from deepchecks.core.errors import DeepchecksValueError
from .detection_formatters import convert_bbox


__all__ = ['ImageInfo', 'numpy_to_image_figure', 'label_bbox_add_to_figure', 'numpy_grayscale_to_heatmap_figure',
           'apply_heatmap_image_properties', 'numpy_to_html_image', 'crop_image']


class ImageInfo:
    """Class with methods defined to extract metadata about image."""

    def __init__(self, img):
        if not isinstance(img, np.ndarray):
            raise DeepchecksValueError('Expect image to be numpy array')
        self.img = img

    def get_size(self) -> t.Tuple[int, int]:
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


def ensure_image(
    image: t.Union[pilimage.Image, np.ndarray, torch.Tensor],
    copy: bool = True
) -> pilimage.Image:
    if isinstance(image, pilimage.Image):
        return image.copy() if copy is True else image
    if isinstance(image, torch.Tensor):
        image = t.cast(np.ndarray, image.numpy())
    if isinstance(image, np.ndarray):
        image = image.squeeze()
        if image.ndim == 3:
            return pilimage.fromarray(image)
        elif image.ndim == 2:
            image = image.astype(np.uint8)
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
    image: t.Union[pilimage.Image, np.ndarray, torch.Tensor],
    bboxes: np.ndarray,
    bbox_notation: t.Optional[str] = None,
    copy_image: bool = True,
    border_width: int = 1,
    color: t.Union[str, t.Dict[np.number, str]] = "red",
) -> pilimage.Image:
    image = ensure_image(image, copy=copy_image)
    
    if bbox_notation is not None:
        bboxes = np.array([
            convert_bbox(
                bbox,
                notation=bbox_notation,
                image_width=image.width,
                image_height=image.height,
                _strict=False
            ).tolist()
            for bbox in bboxes
        ])
        
    draw = pildraw.ImageDraw(image)

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
        draw.text(xy=(x0 +  (w * 0.5), y0 + (h * 0.2)), text=str(clazz), fill=color_to_use)
    
    return image


def display_thumbnails(
    images: t.Union[t.Sequence[pilimage.Image], t.Sequence[np.ndarray]],
    size: t.Optional[t.Tuple[int, int]] = None,
    columns: int = 3,
    copy_image: bool = True,
) -> str:
    if len(images) > 1:
        template = dedent("""
        <div 
            id="thumbnails-container"
            style="
                display: grid; 
                grid-template-columns: repeat({n_of_columns}, 1fr); 
                grid-gap: 10px;
                align-content: space-evenly;
                justify-content: space-evenly;">
            {content}
        </div>
        """)
        image_template = dedent("""
        <img
            src="data:image/png;base64,{img}"
            style="
                justify-self: center;
                align-self: center;"/>
        """)
    else:
        template = dedent("""
        <div 
            id="thumbnails-container"
            style="
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;">
            {content}
        </div>
        """)
        image_template = '<img src="data:image/png;base64,{img}"/>'
    
    tags = []

    for img in images:
        if size is not None:
            img = ensure_image(img, copy=copy_image)
            img.thumbnail(size=size)
        else:
            img = ensure_image(img, copy=False)
        
        img_bytes = io.BytesIO()
        img.save(fp=img_bytes, format="JPEG")
        img_bytes.seek(0)
        img_b64 = base64.b64encode(img_bytes.read()).decode("ascii")
        tags.append(image_template.format(img=img_b64))
        img_bytes.close()
    
    return template.format(content="\n".join(tags), n_of_columns=columns,)


# def numpy_to_html_image(
#     data: np.ndarray,
#     bboxes: t.Optional[torch.Tensor] = None
# ):
#     """Use plotly to create PNG image out of numpy data.

#     Returns
#     ------
#     str
#         HTML img tag with the embedded picture
#     """
#     dimension = data.shape[2]
    
#     if dimension == 1:
#         raise NotImplementedError()
#         fig = go.Figure(go.Heatmap(z=data.squeeze(), colorscale='gray', hoverinfo='skip'))
#         apply_heatmap_image_properties(fig)
#         # fig.update_traces(showscale=False)
#     elif dimension == 3:
#         # fig = go.Figure(go.Image(z=data, hoverinfo='skip'))
#         image = pilimage.fromarray(data)
#     else:
#         raise DeepchecksValueError(f'Don\'t know how to plot images with {dimension} dimensions')

#     if bboxes is not None:
#         draw_bboxes(bboxes, image)

#     # fig.update_yaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
#     # fig.update_xaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
#     # fig.update_layout(margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
#     # png = base64.b64encode(fig.to_image('png')).decode('ascii')
#     buffer = io.BytesIO()
#     image.save(buffer, format='png')
#     buffer.seek(0)
#     png = base64.b64encode(buffer.read()).decode('ascii')
    
#     div = dedent("""
#     <div 
#         id="images-grid"
#         style="
#             display: grid; 
#             grid-template-columns: repeat({n_of_columns}, 1fr); 
#             grid-gap: 10px;
#             align-content: space-evenly;
#             justify-content: space-evenly;">
#         {content}
#     </div>
#     """)
    
#     return dedent(f"""<img src="data:image/png;base64, {png}"/>""")


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


# def draw_bboxes(
#     bboxes: torch.Tensor, 
#     image: pilimage.Image, 
#     color: str = 'red',
#     prediction: bool = False,
#     width: int = 2
# ):
#     for bbox in bboxes:
#         if prediction:
#             x0, y0, w, h, _, clazz = bbox.tolist()
#         else:
#             clazz, x0, y0, w, h = bbox.tolist()
#         x1, y1 = x0 + w, y0 + h
#         draw = pildraw.ImageDraw(image)
#         draw.rectangle(xy=(x0, y0, x1, y1), width=width, outline=color)
#         draw.text(xy=(x0 +  (w * 0.5), y0 + (h * 0.2)), text=str(clazz), fill=color)


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


def crop_image(img: np.ndarray, x, y, w, h) -> np.ndarray:
    """Return the cropped numpy array image by x, y, w, h coordinates (top left corner, width and height."""
    # Convert x, y, w, h to integers if not integers already:
    x, y, w, h = [round(n) for n in [x, y, w, h]]

    # Make sure w, h don't extend the bounding box outside of image dimensions:
    h = min(h, img.shape[0] - y - 1)
    w = min(w, img.shape[1] - x - 1)

    return img[y:y + h, x:x + w]
