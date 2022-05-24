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

import imgaug
import numpy as np
import torch
from IPython.display import HTML, display

from deepchecks.core.errors import ValidationError
from deepchecks.utils.ipython import is_headless, is_notebook
from deepchecks.utils.strings import create_new_file_name
from deepchecks.vision.batch_wrapper import apply_to_tensor
from deepchecks.vision.utils.detection_formatters import DEFAULT_PREDICTION_FORMAT
from deepchecks.vision.utils.image_functions import draw_bboxes, ensure_image, prepare_thumbnail
from deepchecks.vision.vision_data import TaskType, VisionData

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


def validate_extractors(dataset: VisionData, model, device=None, image_save_location: str = None,
                        save_images: bool = True):
    """Validate for given data_loader and model that the extractors are valid.

    Parameters
    ----------
    dataset : VisionData
        the dataset to validate.
    model :
        the model to validate.
    device : torch.device
        device to run model on
    image_save_location : str , default: None
        if location is given and the machine doesn't support GUI,
        the images will be saved there.
    save_images : bool , default: True
        if the machine doesn't support GUI the displayed images will be saved
        if the value is True.
    """
    print('Deepchecks will try to validate the extractors given...')
    batch = apply_to_tensor(next(iter(dataset.data_loader)), lambda it: it.to(device))
    images = None
    labels = None
    predictions = None
    label_formatter_error = None
    image_formatter_error = None
    prediction_formatter_error = None
    device = device or torch.device('cpu')

    try:
        dataset.validate_label(batch)
        labels = dataset.batch_to_labels(batch)
    except ValidationError as ex:
        label_formatter_error = 'Fail! ' + str(ex)
    except Exception:  # pylint: disable=broad-except
        label_formatter_error = 'Got exception \n' + traceback.format_exc()

    try:
        dataset.validate_image_data(batch)
        images = dataset.batch_to_images(batch)
    except ValidationError as ex:
        image_formatter_error = 'Fail! ' + str(ex)
    except Exception:  # pylint: disable=broad-except
        image_formatter_error = 'Got exception \n' + traceback.format_exc()

    try:
        dataset.validate_prediction(batch, model, device)
        predictions = dataset.infer_on_batch(batch, model, device)
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
        image = ensure_image(images[0], copy=False)
        image_title = 'Visual example of an image.'
        if dataset.task_type == TaskType.OBJECT_DETECTION:
            if label_formatter_error is None:
                image = draw_bboxes(image, labels[0], copy_image=False)
            if prediction_formatter_error is None:
                image = draw_bboxes(image, predictions[0], copy_image=False, color='blue',
                                    bbox_notation=DEFAULT_PREDICTION_FORMAT)

            if label_formatter_error is None and prediction_formatter_error is None:
                image_title = 'Visual examples of an image with prediction and label data. Label is red, ' \
                              'prediction is blue, and deepchecks loves you.'
            elif label_formatter_error is None:
                image_title = 'Visual example of an image with label data. Could not display prediction.'
            elif prediction_formatter_error is None:
                image_title = 'Visual example of an image with prediction data. Could not display label.'
            else:
                image_title = 'Visual example of an image. Could not display label or prediction.'
        elif dataset.task_type == TaskType.CLASSIFICATION:
            if label_formatter_error is None:
                image_title += f' Label class {labels[0]}'
            if prediction_formatter_error is None:
                pred_class = predictions[0].argmax()
                image_title += f' Prediction class {pred_class}'
    else:
        image = None
        image_title = None

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

    if image:
        if not is_notebook():
            msg += 'Visual images & label & prediction: should open in a new window'
    else:
        msg += 'Visual images & label & prediction: Unable to show due to invalid image formatter.'

    if is_notebook():
        display(HTML(msg))
        if image:
            image_html = '<div style="display:flex;flex-direction:column;align-items:baseline;">' \
                         f'{prepare_thumbnail(image, size=(200,200))}<p>{image_title}</p></div>'
            display(HTML(image_html))
    else:
        print(msg)
        if image:
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
                    print(image_title)
                    print('validate_extractors can be set to skip the image saving or change the save path')
                    print('*******************************************************************************')
            else:
                image.show()
