# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module for validation of the vision module."""
import random
import traceback
import typing as t
import numpy as np
import torch
import imgaug

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core import errors
from deepchecks import vision  # pylint: disable=unused-import, is used in type annotations
from deepchecks.utils.ipython import is_notebook
from deepchecks.vision.utils.base_formatters import BaseLabelFormatter, BasePredictionFormatter
from deepchecks.vision.utils import ImageFormatter, ClassificationLabelFormatter, DetectionLabelFormatter
from deepchecks.vision.utils.image_functions import numpy_to_image_figure, apply_heatmap_image_properties, \
    label_bbox_add_to_figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from io import BytesIO
from IPython.display import display, HTML


__all__ = ['validate_model', 'set_seeds', 'apply_to_tensor', 'validate_formatters']


def validate_model(dataset: 'vision.VisionData', model: t.Any):
    """Receive a dataset and a model and check if they are compatible.

    Parameters
    ----------
    dataset : VisionData
        Built on a dataloader on which the model can infer.
    model : Any
        Model to be validated

    Raises
    ------
    DeepchecksValueError
        If the dataset and the model are not compatible
    """
    try:
        model(next(iter(dataset.get_data_loader()))[0])
    except Exception as exc:
        raise errors.ModelValidationError(
            f'Got error when trying to predict with model on dataset: {str(exc)}'
        )


def set_seeds(seed: int):
    """Set seeds for reproducibility.

    Imgaug uses numpy's State
    Albumentation uses Python and imgaug seeds

    Parameters
    ----------
    seed : int
        Seed to be set
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        imgaug.seed(seed)


T = t.TypeVar('T')


def apply_to_tensor(
    x: T,
    fn: t.Callable[[torch.Tensor], torch.Tensor]
) -> T:
    """Apply provided function to tensor instances recursivly."""
    if isinstance(x, torch.Tensor):
        return t.cast(T, fn(x))
    elif isinstance(x, (str, bytes, bytearray)):
        return x
    elif isinstance(x, (list, tuple, set)):
        return type(x)(apply_to_tensor(it, fn) for it in x)
    elif isinstance(x, dict):
        return type(x)((k, apply_to_tensor(v, fn)) for k, v in x.items())
    return x


def validate_formatters(data_loader, model, label_formatter: BaseLabelFormatter, image_formatter: ImageFormatter,
                        prediction_formatter: BasePredictionFormatter):
    """Validate for given data_loader and model that the formatters are valid."""
    print('Deepchecks will try to validate the formatters given...')
    batch = next(iter(data_loader))
    images = None
    labels = None
    predictions = None
    label_formatter_error = None
    image_formatter_error = None
    prediction_formatter_error = None

    try:
        label_formatter.validate_label(batch)
        labels = label_formatter(batch)
    except DeepchecksValueError as ex:
        label_formatter_error = str(ex)
    except Exception:  # pylint: disable=broad-except
        label_formatter_error = 'Got exception \n' + traceback.format_exc()

    try:
        image_formatter.validate_data(batch)
        images = image_formatter(batch)
    except DeepchecksValueError as ex:
        image_formatter_error = str(ex)
    except Exception:  # pylint: disable=broad-except
        image_formatter_error = 'Got exception \n' + traceback.format_exc()

    try:
        prediction_formatter.validate_prediction(batch, model, torch.device('cpu'))
        predictions = prediction_formatter(batch, model, torch.device('cpu'))
    except DeepchecksValueError as ex:
        prediction_formatter_error = str(ex)
    except Exception:  # pylint: disable=broad-except
        prediction_formatter_error = 'Got exception \n' + traceback.format_exc()

    # Classes
    if label_formatter_error is None:
        classes = label_formatter.get_classes(labels)
    else:
        classes = None
    # Plot
    if image_formatter_error is None:
        sample_image = images[0]
        if isinstance(label_formatter, DetectionLabelFormatter):
            # In case both label and prediction are valid show image side by side
            if prediction_formatter_error is None and label_formatter_error is None:
                fig = make_subplots(rows=1, cols=2)
                fig.add_trace(numpy_to_image_figure(sample_image), row=1, col=1)
                fig.add_trace(numpy_to_image_figure(sample_image), row=1, col=2)
                label_bbox_add_to_figure(labels[0], fig, row=1, col=1)
                label_bbox_add_to_figure(predictions[0], fig, prediction=True, color='orange', row=1, col=2)
                fig.update_xaxes(title_text='Label', row=1, col=1)
                fig.update_xaxes(title_text='Prediction', row=1, col=2)
                fig.update_layout(title='Visual examples of an image with prediction and label data')
            else:
                fig = go.Figure(numpy_to_image_figure(sample_image))
                # In here only label formatter or prediction formatter are valid (or none of them)
                if label_formatter_error is None:
                    label_bbox_add_to_figure(labels[0], fig)
                    fig.update_xaxes(title='Label')
                    fig.update_layout(title='Visual example of an image with label data')
                elif prediction_formatter_error is None:
                    label_bbox_add_to_figure(predictions[0], fig, prediction=True, color='orange')
                    fig.update_xaxes(title='Prediction')
                    fig.update_layout(title='Visual example of an image with prediction data')

        elif isinstance(label_formatter, ClassificationLabelFormatter):
            fig = go.Figure(numpy_to_image_figure(sample_image))
            # Create figure title
            title = 'Visual example of an image'
            if label_formatter_error is None and prediction_formatter_error is None:
                title += ' with prediction and label data'
            elif label_formatter_error is None:
                title += ' with label data'
            elif prediction_formatter_error is None:
                title += ' with prediction data'
            # Create x-axis title
            x_title = []
            if label_formatter_error is None:
                x_title.append(f'Label: {labels[0]}')
            if prediction_formatter_error is None:
                x_title.append(f'Prediction: {predictions[0]}')

            fig.update_layout(title=title)
            fig.update_xaxes(title=', '.join(x_title))
        else:
            raise DeepchecksValueError(f'Not implemented for label formatter: {type(label_formatter).__name__}')

        if ImageFormatter.get_dimension(sample_image) == 1:
            apply_heatmap_image_properties(fig)
        fig.update_yaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_xaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
    else:
        fig = None

    def get_header(x):
        if is_notebook():
            return f'<h4>{x}</h4>'
        else:
            return x + '\n' + ''.join(['-'] * len(x)) + '\n'

    line_break = '<br>' if is_notebook() else '\n'
    msg = get_header('Structure validation')
    msg += f'Label formatter: {label_formatter_error if label_formatter_error else "Pass!"}{line_break}'
    msg += f'Prediction formatter: {prediction_formatter_error if prediction_formatter_error else "Pass!"}{line_break}'
    msg += f'Image formatter: {image_formatter_error if image_formatter_error else "Pass!"}{line_break}'
    msg += line_break
    msg += get_header('Content validation')
    msg += 'For validating the content within the structure you have to manually observe the classes, image, label ' \
           f'and prediction.{line_break}'

    msg += 'Examples of classes observed in the batch\'s labels: '
    if classes:
        msg += f'{classes[:5]}{line_break}'
    else:
        msg += f'Unable to show due to invalid label formatter.{line_break}'

    if fig:
        if not is_notebook():
            msg += 'Visual images & label & prediction: should open in a new window'
    else:
        msg += 'Visual images & label & prediction: Unable to show due to invalid image formatter.'

    if is_notebook():
        display(HTML(msg))
        if fig:
            display(HTML(fig.to_image('svg').decode('utf-8')))
    else:
        print(msg)
        if fig:
            image = Image.open(BytesIO(fig.to_image('jpg')))
            image.show()
