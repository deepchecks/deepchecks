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
import typing as t
import numpy as np
import torch

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
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    labels = label_formatter(batch[1])
    try:
        label_formatter.validate_label(labels)
        label_formatter_error = None
    except DeepchecksValueError as ex:
        label_formatter_error = str(ex)

    images = image_formatter(batch[0])
    try:
        image_formatter.validate_data(images)
        image_formatter_error = None
    except DeepchecksValueError as ex:
        image_formatter_error = str(ex)

    predictions = prediction_formatter(model(batch[0]))
    try:
        prediction_formatter.validate_prediction(predictions)
        prediction_formatter_error = None
    except DeepchecksValueError as ex:
        prediction_formatter_error = str(ex)

    # Classes
    if label_formatter_error is None:
        if isinstance(label_formatter, DetectionLabelFormatter):
            def get_classes(tensor):
                if len(tensor) == 0:
                    return set()
                return set(tensor[:, 0].tolist())

            classes = list(set().union(*[get_classes(x) for x in labels]))
        elif isinstance(label_formatter, ClassificationLabelFormatter):
            classes = labels.tolist()
        else:
            raise DeepchecksValueError(f'Not implemented for label formatter: {type(label_formatter).__name__}')
    else:
        classes = None
    # Plot
    if image_formatter_error is None:
        sample_image = images[0]
        if isinstance(label_formatter, DetectionLabelFormatter):
            # In case both label and prediction ç are valid show image side by side
            if prediction_formatter_error is None and label_formatter_error is None:
                fig = make_subplots(rows=1, cols=2)
                fig.add_trace(numpy_to_image_figure(sample_image), row=1, col=1)
                fig.add_trace(numpy_to_image_figure(sample_image), row=1, col=2)
                label_bbox_add_to_figure(labels[0], fig, row=1, col=1)
                label_bbox_add_to_figure(predictions[0], fig, prediction=True, color='orange', row=1, col=2)
                fig.update_xaxes(title_text='Label', row=1, col=1)
                fig.update_xaxes(title_text='Prediction', row=1, col=2)
            else:
                fig = go.Figure(numpy_to_image_figure(sample_image))
                # In here only label formatter or prediction formatter are valid (or none of them)
                if label_formatter_error is None:
                    label_bbox_add_to_figure(labels[0], fig)
                    fig.update_xaxes(title='Label')
                elif prediction_formatter_error is None:
                    label_bbox_add_to_figure(predictions[0], fig, prediction=True, color='orange')
                    fig.update_xaxes(title='Prediction')
        elif isinstance(label_formatter, ClassificationLabelFormatter):
            fig = go.Figure(numpy_to_image_figure(sample_image))
            title = ''
            if label_formatter_error is None:
                title += f'Label: {labels[0]} '
            if prediction_formatter_error is None:
                title += f'Prediction: {predictions[0]}'
            fig.update_xaxes(title=title)
        else:
            raise DeepchecksValueError(f'Not implemented for label formatter: {type(label_formatter).__name__}')

        if ImageFormatter.get_dimension(sample_image) == 1:
            apply_heatmap_image_properties(fig)
        fig.update_yaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_xaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_layout(title='Visual examples of image (with prediction and with label data)')
    else:
        fig = None

    msg = 'Structure validation results:\n'
    msg += f'Label formatter: {label_formatter_error if label_formatter_error else "Pass!"}\n'
    msg += f'Prediction formatter: {prediction_formatter_error if prediction_formatter_error else "Pass!"}\n'
    msg += f'Image formatter: {image_formatter_error if image_formatter_error else "Pass!"}\n'
    msg += '\n'
    msg += 'Content validation results:\n'
    msg += 'For validating the content within the structure you have to manually observe the classes, image, label ' \
           'and prediction.\n'
    if classes:
        msg += f'Classes (observed from the batch labels): {classes[:10]}\n'
    else:
        msg += 'Classes: Unable to show due to invalid label formatter.\n'

    msg += 'Visual images & label & prediction'
    if fig:
        if not is_notebook():
            msg += ' should open in a new window'
    else:
        msg += ': Unable to show due to invalid image formatter.'

    if is_notebook():
        display(HTML(msg.replace('\n', '<br>')))
        if fig:
            display(HTML(fig.to_image('svg').decode('utf-8')))
    else:
        print(msg)
        if fig:
            image = Image.open(BytesIO(fig.to_image('jpg')))
            image.show()
