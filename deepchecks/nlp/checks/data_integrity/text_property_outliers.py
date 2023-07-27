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
from typing_extensions import Self

from deepchecks import ConditionCategory, ConditionResult
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import DeepchecksValueError, NotEnoughSamplesError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp.utils.nlp_plot import get_text_outliers_graph
from deepchecks.utils.dataframes import hide_index_for_display
from deepchecks.utils.outliers import iqr_outliers_range, sharp_drop_outliers_range
from deepchecks.utils.strings import format_percent

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
    iqr_scale : float , default : 2
        The scale to multiply the IQR range for the outliers' detection
    sharp_drop_ratio : float, default : 0.9
        The size of the sharp drop to detect categorical outliers
    min_samples : int , default : 10
        Minimum number of samples required to calculate IQR. If there are not enough non-null samples for a specific
        property, the check will skip it. If all properties are skipped, the check will raise a NotEnoughSamplesError.
    """

    def __init__(self,
                 n_show_top: int = 5,
                 iqr_percentiles: t.Tuple[int, int] = (25, 75),
                 iqr_scale: float = 2,
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

        for name, values in properties.items():
            is_numeric = name not in cat_properties

            try:
                if not isinstance(values[0], list):
                    if is_numeric:
                        # Check for non numeric data in the column
                        curr_nan_count = pd.isnull(values).sum()
                        values = pd.to_numeric(values, errors='coerce')
                        updated_nan_count = pd.isnull(values).sum()
                        if updated_nan_count > curr_nan_count:
                            raise DeepchecksValueError('Numeric property contains non-numeric values.')
                    # If the property is single value per sample, then wrap the values in list in order to
                    # work on fixed structure
                    values = [[x] for x in values]

                if is_numeric:
                    values_arr = np.hstack(values).astype(float).squeeze()
                    values_arr = np.array([x for x in values_arr if pd.notnull(x)])
                else:
                    values_arr = np.hstack(values).astype(str).squeeze()

                if len(values_arr) < self.min_samples:
                    raise NotEnoughSamplesError(f'Not enough non-null samples to calculate outliers'
                                                f'(min_samples={self.min_samples}).')

                if is_numeric:
                    lower_limit, upper_limit = iqr_outliers_range(values_arr, self.iqr_percentiles,
                                                                  self.iqr_scale, self.sharp_drop_ratio)
                else:
                    # Counting the frequency of each category. Normalizing because distribution graph shows percentage.
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
                    np.apply_along_axis(lambda i, sort_arr=values_arr: sort_arr[i],
                                        axis=0, arr=bottom_outliers).argsort()
                ]

                text_outliers = np.concatenate([bottom_outliers, top_outliers])

                result[name] = {
                    'indices': [dataset.get_original_text_indexes()[i] for i in text_outliers],
                    # For the upper and lower limits doesn't show values that are smaller/larger than the actual values
                    # we have in the data
                    'lower_limit': max(lower_limit, min(values_arr)),
                    'upper_limit': min(upper_limit, max(values_arr)) if is_numeric else None,
                    'outlier_ratio': len(text_outliers) / len(values_arr),
                }
            except Exception as exp:  # pylint: disable=broad-except
                result[name] = f'{exp}'

        # Create display
        if context.with_display:
            display = []
            no_outliers = pd.Series([], dtype='object')

            # Sort the result map based on the length of indices and if there are any error message associated to
            # any property, keep that property at the very end.
            sorted_result_items = sorted(result.items(),
                                         key=lambda x: len(x[1].get('indices', [])) if isinstance(x[1], dict) else 0,
                                         reverse=True)

            for property_name, info in sorted_result_items:

                # If info is string it means there was error
                if isinstance(info, str):
                    no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=[info])])
                elif len(info['indices']) == 0:
                    no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=['No outliers found.'])])
                else:
                    if len(display) < self.n_show_top:
                        if property_name not in cat_properties:
                            dist = df_properties[property_name].astype(float)
                        else:
                            dist = df_properties[property_name]
                        lower_limit = info['lower_limit']
                        upper_limit = info['upper_limit']

                        try:
                            fig = get_text_outliers_graph(
                                dist=dist,
                                data=dataset.text,
                                lower_limit=lower_limit,
                                upper_limit=upper_limit,
                                dist_name=property_name,
                                is_categorical=property_name in cat_properties
                            )

                            display.append(fig)
                        except Exception as exp:  # pylint: disable=broad-except
                            result[property_name] = f'{exp}'
                            no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=[exp])])
                    else:
                        no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=[
                            f'Outliers found but not shown in graphs (n_show_top={self.n_show_top}).'])])

            if not no_outliers.empty:
                grouped = no_outliers.groupby(level=0).unique().str.join(', ')
                grouped_df = pd.DataFrame(grouped, columns=['Properties'])
                grouped_df['More Info'] = grouped_df.index
                grouped_df = grouped_df[['More Info', 'Properties']]
                display.append('<h5><b>Properties Not Shown:</h5></b>')
                display.append(hide_index_for_display(grouped_df))

        else:
            display = None

        return CheckResult(result, display=display)

    def add_condition_outlier_ratio_less_or_equal(self: Self, threshold: float = 0.05,
                                                  properties_to_ignore: t.Optional[t.List[str]] = None) -> Self:
        """Add condition - outlier ratio in every property is less or equal to ratio.

        Parameters
        ----------
        threshold : float , default: 0.05
            Maximum threshold of outliers ratio per property.
        properties_to_ignore : t.Optional[t.List[str]] , default: None
            List of properties to ignore for the condition.
        """

        def condition(result: t.Dict[str, t.Any]):
            failed_properties = []
            worst_property = ''
            worst_ratio = 0

            for property_name, info in result.items():
                if properties_to_ignore is not None and property_name in properties_to_ignore:
                    continue
                if isinstance(info, str):
                    continue
                if info['outlier_ratio'] > threshold:
                    failed_properties.append(property_name)
                if info['outlier_ratio'] > worst_ratio:
                    worst_property = property_name
                    worst_ratio = info['outlier_ratio']

            if len(failed_properties) > 0:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found {len(failed_properties)} properties with outlier ratios above threshold.'
                                       f'</br>Property with highest ratio is {worst_property} with outlier ratio of '
                                       f'{format_percent(worst_ratio)}')
            else:
                return ConditionResult(ConditionCategory.PASS,
                                       f'All properties have outlier ratios below threshold. '
                                       f'Property with highest ratio is {worst_property} with outlier ratio of'
                                       f' {format_percent(worst_ratio)}')

        return self.add_condition(f'Outlier ratio in all properties is less or equal than {format_percent(threshold)}',
                                  condition)
