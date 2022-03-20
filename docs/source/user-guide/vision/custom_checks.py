"""
.. custom_checks:

.. currentmodule:: deepchecks.vision.base_checks

=============================================
Writing Custom Computer Vision Checks
=============================================

Deepchecks offers a wide range of checks for computer vision problems, addressing distribution issues, performance
checks and more. Nevertheless, in order to fully validate your ML pipeline you often need to write your own checks.
This guide will walk you through the basics of writing your own checks.

We recommend writing a single check for each aspect of the model or data you would like to validate. As explained in
:doc:`/user-guide/general/deepchecks_hierarchy`, the role of the check is to run the logic and output a display and
a pythonic value. Then, a condition can be defined on that value to determine if the check is successful or not.

#. :ref:`Vision Checks Structure`_
#. :ref:`Write a Basic Check`_
#. :ref:`Check Display`_
#. :ref:`Defining a Condition`_

Vision Checks Structure
========================

Vision checks all inherit from one of the following classes:

- :class:`~deepchecks.vision.base_checks.SingleDatasetCheck` - Check that runs on a single dataset and model.
- :class:`~deepchecks.vision.base_checks.TrainTestCheck` - Check that runs on a train and test dataset and model.
- :class:`~deepchecks.vision.base_checks.ModelOnlyCheck` - Check that runs on only a model .

All three classes inherit from the :class:`~deepchecks.core.checks.BaseCheck` BaseCheck, same as checks in any other
deepchecks subpackage.

The first two classes of checks run some logic on the image data, and so the check structure is designed to enable
accumulating and computation on batches outputted by the dataloader. These first two types of checks must implement
the following three methods:

- initialize_run - Actions to be performed before starting to iterate over the dataloader batches.
- update - Actions to be performed on each batch.
- compute - Actions to be performed after iterating over all the batches. Returns the check display and the return
value.

While `ModelOnlyCheck` alone do not implement the update method. Apart from that, the check init should recipient and
handle check parameters.


Write a Basic Check
========================

Let's implement a simple check, comparing the average of each color channel between the train and the test dataset.

We'll start by writing the simplest possible example, returning only a dict of the color averages.

*Good to know: the return value of a check can be any object, a number, dictionary, string, etc…*
"""
from deepchecks.vision.base_checks import TrainTestCheck
from deepchecks.vision.context import Context, Batch
from deepchecks.core.checks import DatasetKind
from deepchecks.core.check_result import CheckResult
import typing as t
import numpy as np


class ColorAveragesCheck(TrainTestCheck):
    """Check if the average of each color channel is the same between the train and test dataset."""

    def __init__(self, channel_names: t.Tuple[str] = None):
        """Init the check and enable customization of the channel_names."""
        super().__init__()
        if channel_names is None:
            self.channel_names = ('R', 'G', 'B')

    def initialize_run(self, context: Context):
        """Initialize the color_averages dict and pixel counter dict."""
        self._color_averages = {
            DatasetKind.TRAIN.value: np.zeros((3,), dtype=np.float64),
            DatasetKind.TEST.value: np.zeros((3,), dtype=np.float64),
        }
        self._pixel_count = {
            DatasetKind.TRAIN.value: 0,
            DatasetKind.TEST.value: 0,
        }

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Add the batch color counts to the color_averages dict, and update counter."""
        images = batch.images
        color_sums = sum(image.sum(axis=(0, 1)) for image in images)  # sum over the batch and pixels
        self._color_averages[dataset_kind.value] += color_sums  # add to the color_averages dict
        # count the number of pixels we have summed
        self._pixel_count[dataset_kind.value] += sum((image.shape[0] * image.shape[1] for image in images))

    def compute(self, context: Context):
        """Compute the color averages and return them."""
        # Divide by the number of pixels to get the average pixel value per color channel
        for dataset_kind in DatasetKind:
            self._color_averages[dataset_kind.value] /= self._pixel_count[dataset_kind.value]
        # Return the color averages in a dict by channel name
        return_value = {d_kind: dict(zip(self.channel_names, color_averages))
                        for d_kind, color_averages in self._color_averages.items()}
        return CheckResult(return_value)

#%%
# Hooray! we just implemented a custom check. To read more about the internal objects  Let's run it and see what happens:

from deepchecks.vision.datasets.detection.coco import load_dataset

train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')

result = ColorAveragesCheck().run(train_ds, test_ds)
result

#%%
# Our check ran successfully, but we got the print “Nothing found”. This is because we haven’t defined to the check
# anything to display, so the default behavior is to print “Nothing found”. In order to access the value that we have
# defined earlier we can use the “value” property on the result.

result.value

#TODO: how does this LINK work?

# %% To see code references for more complex checks (that can receive parameters etc.), check out any of your
# favorite checks from our API Reference (LINK).

#%%
# Check Display
# ========================
#
# Most of the times we will want our checks to have a visual display that will quickly summarize the check result. We
# can pass objects for display to the CheckResult. Objects for display should be of type: html string, dataframe or a
# function that plots a graph. Let’s define a graph that will be displayed using `Plotly <https://plotly.com/>`_. We
# will inherit from the original check to shorten the code an update only the compute method.
#
# *Good to know: ``display`` can receive a single object to display or a list of objects*

import pandas as pd
import plotly.express as px


class ColorAveragesHistCheck(ColorAveragesCheck):

    def compute(self, context: Context):
        """Compute the color averages and return them. Also display a histogram comparing train and test."""
        # Divide by the number of pixels to get the average pixel value per color channel
        for dataset_kind in DatasetKind:
            self._color_averages[dataset_kind.value] /= self._pixel_count[dataset_kind.value]
        # Return the color averages in a dict by channel name
        return_value = {d_kind: dict(zip(self.channel_names, color_averages))
                        for d_kind, color_averages in self._color_averages.items()}

        # Display a histogram comparing train and test
        color_averages_df = pd.DataFrame(return_value).unstack().reset_index()
        color_averages_df.columns = ['Dataset', 'Channel', 'Pixel Value']
        fig = px.histogram(color_averages_df, x='Dataset', y='Pixel Value', color='Channel', barmode='group',
                           histfunc='avg', color_discrete_sequence=['rgb(255,0,0)', 'rgb(0,255,0)', 'rgb(0,0,255)'],
                           title='Color Averages Histogram')
        return CheckResult(return_value, display=[fig])

#%%
# Let check it out:

result = ColorAveragesHistCheck().run(train_ds, test_ds)
result

# %%
# Voilà! Now we have a check that prints a graph and has a value. We can add this check
# to any Suite, and it will run within it.

# %%
# Defining a Condition
# ========================
#
# Finally, we can add a condition to our check. A condition is a function that receives the result of the check and
# returns a condition result object. To read more on conditions, check out user guide (LINK). # TODO: how to link to configure a check condition user guide?
# In this case, we'll define a condition verifying that the color averages haven't changed by more than 10%.

from deepchecks.core import ConditionResult


class ColorAveragesHistConditionCheck(ColorAveragesHistCheck):

    def add_condition_color_average_change_not_greater_than(self, change_ratio: float = 0.1) -> ConditionResult:
        """Add a condition verifying that the color averages haven't changed by more than change_ratio%."""

        def condition(check_result: CheckResult) -> ConditionResult:
            failing_channels = []
            # Iterate over the color averages and verify that they haven't changed by more than change_ratio
            for channel in check_result.value[DatasetKind.TRAIN.value].keys():
                if abs(check_result.value[DatasetKind.TRAIN.value][channel] -
                       check_result.value[DatasetKind.TEST.value][channel]) > change_ratio:
                    failing_channels.append(channel)

            # If there are failing channels, return a condition result with the failing channels
            if failing_channels:
                return ConditionResult(False, f'The color averages have changes by more than threshold in the channels'
                                              f' {failing_channels}.')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Change in color averages not greater than {change_ratio:.2%}', condition)

#%%
# Let check it out:
result = ColorAveragesHistConditionCheck().run(train_ds, test_ds)
result

#%%
# And now our check we will alert us automatically if the color averages have changed by more than 10%!
