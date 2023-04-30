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
import typing as t

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import NotEnoughSamplesError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp.utils.nlp_plot import get_text_outliers_graph
from deepchecks.utils.outliers import iqr_outliers_range, sharp_drop_outliers_range

__all__ = ['TextPropertyOutliers']


class TextPropertyOutliers(SingleDatasetCheck):
    """Find outliers with respect to the given properties.

    The check finds outliers in the text properties.
    For numeric properties, the check uses `IQR <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to
    detect outliers out of the single dimension properties.
    For categorical properties, the check searches for a relative "sharp drop" in values in order to detect outliers.

    Parameters
    ----------
    n_show_top : int , default : 5
        number of graphs to show (ordered from the property with the most outliers to the least)
    iqr_percentiles : Tuple[int, int] , default : (25, 75)
        Two percentiles which define the IQR range
    iqr_scale : float , default : 1.5
        The scale to multiply the IQR range for the outliers detection
    sharp_drop_ratio : float, default : 0.9
        The size of the sharp drop to detect categorical outliers
    min_samples : int , default : 10
        Minimum number of samples required to calculate IQR. If there are not enough non-null samples a specific
        property, the check will skip it. If all properties are skipped, the check will raise a NotEnoughSamplesError.
    """

    def __init__(self,
                 n_show_top: int = 5,
                 iqr_percentiles: t.Tuple[int, int] = (25, 75),
                 iqr_scale: float = 1.5,
                 sharp_drop_ratio: float = 0.9,
                 min_samples: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.iqr_percentiles = iqr_percentiles
        self.iqr_scale = iqr_scale
        self.sharp_drop_ratio = sharp_drop_ratio
        self.n_show_top = n_show_top
        self.min_samples = min_samples

    def run_logic(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Compute final result."""
        dataset = context.get_data_by_kind(dataset_kind)
        result = {}

        df_properties = dataset.properties
        cat_properties = dataset.categorical_properties
        properties = df_properties.to_dict(orient='list')

        if all(len(np.hstack(v).squeeze()) < self.min_samples for v in properties.values()):
            raise NotEnoughSamplesError(f'Need at least {self.min_samples} non-null samples to calculate outliers.')

        # The values are in the same order as the batch order, so always keeps the same order in order to access
        # the original sample at this index location
        for name, values in properties.items():
            # If the property is single value per sample, then wrap the values in list in order to work on fixed
            # structure
            if not isinstance(values[0], list):
                values = [[x] for x in values]

            is_numeric = name not in cat_properties

            if is_numeric:
                values_arr = np.hstack(values).astype(float).squeeze()
                values_arr = np.array([x for x in values_arr if pd.notnull(x)])
            else:
                values_arr = np.hstack(values).astype(str).squeeze()

            if len(values_arr) < self.min_samples:
                result[name] = 'Not enough non-null samples to calculate outliers.'
                continue

            if is_numeric:
                lower_limit, upper_limit = iqr_outliers_range(values_arr, self.iqr_percentiles, self.iqr_scale)
            else:
                # Counting the frequency of each category. Normalizing because distribution graph shows the percentage.
                counts_map = pd.Series(values_arr.astype(str)).value_counts(normalize=True).to_dict()
                lower_limit = sharp_drop_outliers_range(sorted(list(counts_map.values()), reverse=True),
                                                        self.sharp_drop_ratio) or 0
                upper_limit = len(values_arr)  # No upper limit for categorical properties
                values_arr = np.array([counts_map[x] for x in values_arr])  # replace the values with the counts

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
                'indices': [dataset.get_original_text_indexes()[i] for i in text_outliers],
                # For the upper and lower limits doesn't show values that are smaller/larger than the actual values
                # we have in the data
                'lower_limit': max(lower_limit, min(values_arr)),
                'upper_limit': min(upper_limit, max(values_arr)) if is_numeric else None,
            }

        # Create display
        if context.with_display:
            display = []
            no_outliers = pd.Series([], dtype='object')

            sorted_result_items = sorted(result.items(), key=lambda x: len(x[1]['indices']), reverse=True)

            for property_name, info in sorted_result_items:
                # If info is string it means there was error
                if isinstance(info, str):
                    no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=[info])])
                elif len(info['indices']) == 0:
                    no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=['No outliers found.'])])
                else:
                    if len(display) < self.n_show_top:
                        dist = df_properties[property_name]
                        lower_limit = info['lower_limit']
                        upper_limit = info['upper_limit']

                        fig = get_text_outliers_graph(
                            dist=dist,
                            data=dataset.text,
                            lower_limit=lower_limit,
                            upper_limit=upper_limit,
                            dist_name=property_name,
                            is_categorical=property_name in cat_properties
                        )

                        display.append(fig)
                    else:
                        no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=[
                            f'Outliers found but not shown in graphs (n_show_top={self.n_show_top}).'])])

            if not no_outliers.empty:
                grouped = no_outliers.groupby(level=0).unique().str.join(', ')
                grouped_df = pd.DataFrame(grouped, columns=['Properties'])
                grouped_df['More Info'] = grouped_df.index
                grouped_df = grouped_df[['More Info', 'Properties']]
                display.append('<h5><b>Properties Not Shown:</h5></b>')
                display.append(grouped_df.style.hide(axis='index') if hasattr(grouped_df.style, 'hide') else
                               grouped_df.style.hide_index())

        else:
            display = None

        return CheckResult(result, display=display)
