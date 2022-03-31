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
import os
import random
import traceback
import numpy as np
import torch
import imgaug

from deepchecks.core.errors import DeepchecksValueError, ValidationError
from deepchecks.utils.ipython import is_headless, is_notebook
from deepchecks.utils.strings import create_new_file_name
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.utils.image_functions import numpy_to_image_figure, label_bbox_add_to_figure
from deepchecks.vision.vision_data import VisionData

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from io import BytesIO
from IPython.display import display, HTML


__all__ = ['set_seeds', 'validate_extractors']


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


def validate_extractors(dataset: VisionData, model, image_save_location: str = None, save_images: bool = True):
    """Validate for given data_loader and model that the extractors are valid.

    Parameters
    ----------
    dataset : VisionData
        the dataset to validate.
    model :
        the model to validate.
    image_save_location : str , default: None
        if location is given and the machine doesn't support GUI,
        the images will be saved there.
    save_images : bool , default: True
        if the machine doesn't support GUI the displayed images will be saved
        if the value is True.
    """
    print('Deepchecks will try to validate the extractors given...')
    batch = next(iter(dataset.data_loader))
    images = None
    labels = None
    predictions = None
    label_formatter_error = None
    image_formatter_error = None
    prediction_formatter_error = None

    try:
        dataset.validate_label(batch)
        labels = dataset.batch_to_labels(batch)
    except ValidationError as ex:
        label_formatter_error = str(ex)
    except Exception:  # pylint: disable=broad-except
        label_formatter_error = 'Got exception \n' + traceback.format_exc()

    try:
        dataset.validate_image_data(batch)
        images = dataset.batch_to_images(batch)
    except ValidationError as ex:
        image_formatter_error = str(ex)
    except Exception:  # pylint: disable=broad-except
        image_formatter_error = 'Got exception \n' + traceback.format_exc()

    try:
        dataset.validate_prediction(batch, model, torch.device('cpu'))
        predictions = dataset.infer_on_batch(batch, model, torch.device('cpu'))
    except ValidationError as ex:
        prediction_formatter_error = str(ex)
    except Exception:  # pylint: disable=broad-except
        prediction_formatter_error = 'Got exception \n' + traceback.format_exc()

    # Classes
    if label_formatter_error is None:
        classes = dataset.get_classes(labels)
    else:
        classes = None
    # Plot
    if image_formatter_error is None:
        sample_image = images[0]
        if dataset.task_type == TaskType.OBJECT_DETECTION:
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

        elif dataset.task_type == TaskType.CLASSIFICATION:
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
            fig = go.Figure(numpy_to_image_figure(sample_image))
            fig.update_layout(title='Visual example of an image')

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
            if is_headless():
                if save_images:
                    if image_save_location is None:
                        save_loc = os.getcwd()
                    else:
                        save_loc = image_save_location
                    full_image_path = os.path.join(save_loc, 'deepchecks_formatted_image.jpg')
                    full_image_path = create_new_file_name(full_image_path)
                    image.save(full_image_path)
                    print('*******************************************************************************')
                    print('This machine does not support GUI')
                    print('The formatted image was saved in:')
                    print(full_image_path)
                    print('validate_extractors can be set to skip the image saving or change the save path')
                    print('*******************************************************************************')
            else:
                image.show()
