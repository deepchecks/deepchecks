# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module of TextPropertyOutliers check."""
import string
import typing as t
from collections import defaultdict
from numbers import Number
from secrets import choice

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import DeepchecksProcessError, NotEnoughSamplesError
from deepchecks.nlp import Context, SingleDatasetCheck, TextData
from deepchecks.nlp.utils.text_utils import trim
from deepchecks.utils.distribution.plot import feature_distribution_traces, get_property_outlier_graph
from deepchecks.utils.outliers import iqr_outliers_range
from deepchecks.utils.strings import format_number

__all__ = ['TextPropertyOutliers']

THUMBNAIL_SIZE = (200, 200)


class TextPropertyOutliers(SingleDatasetCheck):
    """Find outliers samples with respect to the given properties.

    The check computes several properties and then computes the number of outliers for each property.
    The check uses `IQR <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to detect outliers out of the
    single dimension properties.

    Parameters
    ----------
    properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        - ``'class_id'`` - for properties that return the class_id. This is used because these
          properties are later matched with the ``VisionData.label_map``, if one was given.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    n_show_top : int , default: 5
        number of outliers to show from each direction (upper limit and bottom limit)
    iqr_percentiles: Tuple[int, int], default: (25, 75)
        Two percentiles which define the IQR range
    iqr_scale: float, default: 1.5
        The scale to multiply the IQR range for the outliers detection
    """
    """Find outliers images with respect to the given properties.

    The check computes several image properties and then computes the number of outliers for each property.
    The check uses `IQR <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to detect outliers out of the
    single dimension properties.

    Parameters
    ----------
    text_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    n_show_top : int , default: 5
        number of outliers to show from each direction (upper limit and bottom limit)
    iqr_percentiles: Tuple[int, int], default: (25, 75)
        Two percentiles which define the IQR range
    iqr_scale: float, default: 1.5
        The scale to multiply the IQR range for the outliers detection
    """

    def __init__(self,
                 properties_list: t.List[t.Dict[str, t.Any]] = None,
                 n_show_top: int = 5,
                 iqr_percentiles: t.Tuple[int, int] = (25, 75),
                 iqr_scale: float = 1.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.properties_list = properties_list
        self.iqr_percentiles = iqr_percentiles
        self.iqr_scale = iqr_scale
        self.n_show_top = n_show_top

    def run_logic(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Compute final result."""
        dataset = context.get_data_by_kind(dataset_kind)
        corpus = defaultdict(list)
        result = {}

        property_types = dataset.properties_types
        df_properties = dataset.properties[[col for col in dataset.properties.columns if property_types[col] == 'numeric']]
        properties = df_properties.to_dict(orient='list')

        # The values are in the same order as the batch order, so always keeps the same order in order to access
        # the original sample at this index location
        for name, values in properties.items():
            # If the property is single value per sample, then wrap the values in list in order to work on fixed
            # structure
            if not isinstance(values[0], list):
                values = [[x] for x in values]

            values_arr = np.hstack(values).astype(float)

            try:
                lower_limit, upper_limit = iqr_outliers_range(values_arr, self.iqr_percentiles, self.iqr_scale)
            except NotEnoughSamplesError:
                result[name] = 'Not enough non-null samples to calculate outliers.'
                continue

            # Get the indices of the outliers
            top_outliers = np.argwhere(values_arr > upper_limit).squeeze(axis=1)
            # Sort the indices of the outliers by the original values
            top_outliers = top_outliers[
                np.apply_along_axis(lambda i, sort_arr=values_arr: sort_arr[i], axis=0, arr=top_outliers).argsort()
            ]

            # Doing the same for bottom outliers
            bottom_outliers = np.argwhere(values_arr < lower_limit).squeeze(axis=1)
            # Sort the indices of the outliers by the original values
            bottom_outliers = bottom_outliers[
                np.apply_along_axis(lambda i, sort_arr=values_arr: sort_arr[i], axis=0, arr=bottom_outliers).argsort()
            ]

            text_outliers = np.concatenate([bottom_outliers, top_outliers])

            result[name] = {
                'indices': [dataset.index[i] for i in text_outliers],
                # For the upper and lower limits doesn't show values that are smaller/larger than the actual values
                # we have in the data
                'lower_limit': max(lower_limit, min(values_arr)),
                'upper_limit': min(upper_limit, max(values_arr)),
            }

        # Create display
        if context.with_display:
            display = []
            no_outliers = pd.Series([], dtype='object')

            for property_name, info in result.items():
                # If info is string it means there was error
                if isinstance(info, str):
                    no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=[info])])
                elif len(info['indices']) == 0:
                    no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=['No outliers found.'])])
                else:
                    dist = df_properties[property_name]
                    lower_limit = info['lower_limit']
                    upper_limit = info['upper_limit']

                    fig = get_property_outlier_graph(dist, dataset.text, lower_limit, upper_limit, property_name)

                    display.append(fig)

            if not no_outliers.empty:
                grouped = no_outliers.groupby(level=0).unique().str.join(', ')
                grouped_df = pd.DataFrame(grouped, columns=['Properties'])
                grouped_df['More Info'] = grouped_df.index
                grouped_df = grouped_df[['More Info', 'Properties']]
                display.append('<h5><b>Properties With No Outliers Found</h5></b>')
                display.append(grouped_df.style.hide(axis='index') if hasattr(grouped_df.style, 'hide') else
                               grouped_df.style.hide_index())

        else:
            display = None

        return CheckResult(result, display=display)
