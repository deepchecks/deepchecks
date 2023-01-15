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
"""Common utilities for distribution checks."""
from numbers import Number
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, wasserstein_distance

from deepchecks.core import ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError, NotEnoughSamplesError
from deepchecks.utils.dict_funcs import get_dict_entry_by_value
from deepchecks.utils.distribution.plot import (CategoriesSortingKind, drift_score_bar_traces,
                                                feature_distribution_traces)
from deepchecks.utils.distribution.preprocessing import preprocess_2_cat_cols_to_same_bins
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES
from deepchecks.utils.strings import format_number

__all__ = ['calc_drift_and_plot', 'get_drift_method', 'SUPPORTED_CATEGORICAL_METHODS', 'SUPPORTED_NUMERIC_METHODS',
           'drift_condition', 'get_drift_plot_sidenote']

PSI_MIN_PERCENTAGE = 0.01
SUPPORTED_CATEGORICAL_METHODS = ['Cramer\'s V', 'PSI']
SUPPORTED_NUMERIC_METHODS = ['Earth Mover\'s Distance']


def get_drift_method(result_dict: Dict):
    """Return which drift scoring methods were in use.

    Parameters
    ----------
    result_dict : Dict
        the result dict of the drift check.
    Returns
    -------
    Tuple(str, str)
        the categorical scoring method and then the numeric scoring method.

    """
    result_df = pd.DataFrame(result_dict).T
    cat_mthod_arr = result_df[result_df['Method'].isin(SUPPORTED_CATEGORICAL_METHODS)]['Method']
    cat_method = cat_mthod_arr.iloc[0] if len(cat_mthod_arr) else None

    num_mthod_arr = result_df[result_df['Method'].isin(SUPPORTED_NUMERIC_METHODS)]['Method']
    num_method = num_mthod_arr.iloc[0] if len(num_mthod_arr) else None

    return cat_method, num_method


def cramers_v(dist1: Union[np.ndarray, pd.Series], dist2: Union[np.ndarray, pd.Series],
              min_category_size_ratio: float = 0, max_num_categories: int = None,
              sort_by: str = 'dist1') -> float:
    """Calculate the Cramer's V statistic.

    For more on Cramer's V, see https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Uses the Cramer's V bias correction, see http://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf

    Function is for categorical data only.

    Parameters
    ----------
    dist1 : Union[np.ndarray, pd.Series]
        array of numerical values.
    dist2 : Union[np.ndarray, pd.Series]
        array of numerical values to compare dist1 to.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category.
    max_num_categories: int, default: None
        max number of allowed categories. If there are more categories than this number, categories are ordered by
        magnitude and all the smaller categories are binned into an "Other" category.
        If max_num_categories=None, there is no limit.
        > Note that if this parameter is used, the ordering of categories (and by extension, the decision which
        categories are kept by name and which are binned to the "Other" category) is done by default according to the
        values of dist1, which is treated as the "expected" distribution. This behavior can be changed by using the
        sort_by parameter.
    sort_by: str, default: 'dist1'
        Specify how categories should be sorted, affecting which categories will get into the "Other" category.
        Possible values:
        - 'dist1': Sort by the largest dist1 categories.
        - 'dist2': Sort by the largest dist2 categories.
        - 'difference': Sort by the largest difference between categories.
        > Note that this parameter has no effect if max_num_categories = None or there are not enough unique categories.
    Returns
    -------
    float
        the bias-corrected Cramer's V value of the 2 distributions.

    """
    dist1_counts, dist2_counts, _ = preprocess_2_cat_cols_to_same_bins(dist1, dist2, min_category_size_ratio,
                                                                       max_num_categories, sort_by)
    contingency_matrix = pd.DataFrame([dist1_counts, dist2_counts])

    # If columns have the same single value in both (causing division by 0), return 0 drift score:
    if contingency_matrix.shape[1] == 1:
        return 0

    # This is based on
    # https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792 # noqa: SC100
    # and reused in other sources
    # (https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9) # noqa: SC100
    chi2 = chi2_contingency(contingency_matrix)[0]
    n = contingency_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def psi(dist1: Union[np.ndarray, pd.Series], dist2: Union[np.ndarray, pd.Series],
        min_category_size_ratio: float = 0, max_num_categories: int = None, sort_by: str = 'dist1') -> float:
    """
    Calculate the PSI (Population Stability Index).

    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf

    Parameters
    ----------
    dist1 : Union[np.ndarray, pd.Series]
        array of numerical values.
    dist2 : Union[np.ndarray, pd.Series]
        array of numerical values to compare dist1 to.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category.
    max_num_categories: int, default: None
        max number of allowed categories. If there are more categories than this number, categories are ordered by
        magnitude and all the smaller categories are binned into an "Other" category.
        If max_num_categories=None, there is no limit.
        > Note that if this parameter is used, the ordering of categories (and by extension, the decision which
        categories are kept by name and which are binned to the "Other" category) is done by default according to the
        values of dist1, which is treated as the "expected" distribution. This behavior can be changed by using the
        sort_by parameter.
    sort_by: str, default: 'dist1'
        Specify how categories should be sorted, affecting which categories will get into the "Other" category.
        Possible values:
        - 'dist1': Sort by the largest dist1 categories.
        - 'dist2': Sort by the largest dist2 categories.
        - 'difference': Sort by the largest difference between categories.
        > Note that this parameter has no effect if max_num_categories = None or there are not enough unique categories.
    Returns
    -------
    psi
        The PSI score
    """
    expected_counts, actual_counts, _ = preprocess_2_cat_cols_to_same_bins(dist1, dist2, min_category_size_ratio,
                                                                           max_num_categories, sort_by)
    size_expected, size_actual = sum(expected_counts), sum(actual_counts)
    psi_value = 0
    for i in range(len(expected_counts)):
        # In order for the value not to diverge, we cap our min percentage value
        e_perc = max(expected_counts[i] / size_expected, PSI_MIN_PERCENTAGE)
        a_perc = max(actual_counts[i] / size_actual, PSI_MIN_PERCENTAGE)
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        psi_value += value

    return psi_value


def earth_movers_distance(dist1: Union[np.ndarray, pd.Series], dist2: Union[np.ndarray, pd.Series],
                          margin_quantile_filter: float):
    """
    Calculate the Earth Movers Distance (Wasserstein distance).

    See https://en.wikipedia.org/wiki/Wasserstein_metric

    Function is for numerical data only.

    Parameters
    ----------
    dist1: Union[np.ndarray, pd.Series]
        array of numerical values.
    dist2: Union[np.ndarray, pd.Series]
        array of numerical values to compare dist1 to.
    margin_quantile_filter: float
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    Returns
    -------
    Any
        the Wasserstein distance between the two distributions.

    Raises
    -------
    DeepchecksValueError
        if the value of margin_quantile_filter is not in range [0, 0.5)

    """
    if not isinstance(margin_quantile_filter, Number) or margin_quantile_filter < 0 or margin_quantile_filter >= 0.5:
        raise DeepchecksValueError(
            f'margin_quantile_filter expected a value in range [0, 0.5), instead got {margin_quantile_filter}')

    if margin_quantile_filter != 0:
        dist1_qt_min, dist1_qt_max = np.quantile(dist1, [margin_quantile_filter, 1 - margin_quantile_filter])
        dist2_qt_min, dist2_qt_max = np.quantile(dist2, [margin_quantile_filter, 1 - margin_quantile_filter])
        dist1 = dist1[(dist1_qt_max >= dist1) & (dist1 >= dist1_qt_min)]
        dist2 = dist2[(dist2_qt_max >= dist2) & (dist2 >= dist2_qt_min)]

    val_max = np.max([np.max(dist1), np.max(dist2)])
    val_min = np.min([np.min(dist1), np.min(dist2)])

    if val_max == val_min:
        return 0

    # Scale the distribution between 0 and 1:
    dist1 = (dist1 - val_min) / (val_max - val_min)
    dist2 = (dist2 - val_min) / (val_max - val_min)

    return wasserstein_distance(dist1, dist2)


def calc_drift_and_plot(train_column: pd.Series,
                        test_column: pd.Series,
                        value_name: str,
                        column_type: str,
                        plot_title: Optional[str] = None,
                        margin_quantile_filter: float = 0.025,
                        max_num_categories_for_drift: int = None,
                        min_category_size_ratio: float = 0.01,
                        max_num_categories_for_display: int = 10,
                        show_categories_by: CategoriesSortingKind = 'largest_difference',
                        categorical_drift_method: str = 'cramer_v',
                        ignore_na: bool = True,
                        min_samples: int = 10,
                        with_display: bool = True,
                        dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES
                        ) -> Tuple[float, str, Optional[Figure]]:
    """
    Calculate drift score per column.

    Parameters
    ----------
    train_column: pd.Series
        column from train dataset
    test_column: pd.Series
        same column from test dataset
    value_name: str
        title of the x axis, if plot_title is None then also the title of the whole plot.
    column_type: str
        type of column (either "numerical" or "categorical")
    plot_title: str or None
        if None use value_name as title otherwise use this.
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category.
    max_num_categories_for_drift: int, default: None
        Max number of allowed categories. If there are more, they are binned into an "Other" category.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:
        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.
    categorical_drift_method: str, default: "cramer_v"
        decides which method to use on categorical variables. Possible values are:
        "cramer_v" for Cramer's V, "PSI" for Population Stability Index (PSI).
    ignore_na: bool, default True
        For categorical columns only. If True, ignores nones for categorical drift. If False, considers none as a
        separate category. For numerical columns we always ignore nones.
    min_samples: int, default: 10
        Minimum number of samples for each column in order to calculate draft
    with_display: bool, default: True
        flag that determines if function will calculate display.
    dataset_names: tuple, default: DEFAULT_DATASET_NAMES
        The names to show in the display for the first and second datasets.
    Returns
    -------
    Tuple[float, str, Callable]
        drift score of the difference between the two columns' distributions (Earth movers distance for
        numerical, PSI for categorical)
        graph comparing the two distributions (density for numerical, stack bar for categorical)
    """
    if min_category_size_ratio < 0 or min_category_size_ratio > 1:
        raise DeepchecksValueError(
            f'min_category_size_ratio expected a value in range [0, 1], instead got {min_category_size_ratio}.')

    if column_type == 'categorical' and ignore_na is False:
        train_dist = np.array(train_column.values).reshape(-1)
        test_dist = np.array(test_column.values).reshape(-1)
    else:
        train_dist = np.array(train_column.dropna().values).reshape(-1)
        test_dist = np.array(test_column.dropna().values).reshape(-1)

    if len(train_dist) < min_samples or len(test_dist) < min_samples:
        raise NotEnoughSamplesError(f'For drift calculations a minimum of {min_samples} samples are needed but '
                                    f'got {len(train_dist)} for train and {len(test_dist)} for test')

    if column_type == 'numerical':
        scorer_name = 'Earth Mover\'s Distance'

        train_dist = train_dist.astype('float')
        test_dist = test_dist.astype('float')

        score = earth_movers_distance(dist1=train_dist, dist2=test_dist, margin_quantile_filter=margin_quantile_filter)

        if not with_display:
            return score, scorer_name, None

        bar_traces, bar_x_axis, bar_y_axis = drift_score_bar_traces(score)
        dist_traces, dist_x_axis, dist_y_axis = feature_distribution_traces(train_dist, test_dist, value_name,
                                                                            dataset_names=dataset_names)

    elif column_type == 'categorical':
        sort_by = 'difference' if show_categories_by == 'largest_difference' else \
            ('dist1' if show_categories_by == 'train_largest' else 'dist2')
        if categorical_drift_method.lower() in ['cramer_v', 'cramers_v']:
            scorer_name = 'Cramer\'s V'
            score = cramers_v(train_dist, test_dist, min_category_size_ratio, max_num_categories_for_drift, sort_by)
        elif categorical_drift_method.lower() == 'psi':
            scorer_name = 'PSI'
            score = psi(train_dist, test_dist, min_category_size_ratio, max_num_categories_for_drift, sort_by)
        else:
            raise ValueError('Expected categorical_drift_method to be one '
                             f'of [cramer_v, PSI], received: {categorical_drift_method}')

        if not with_display:
            return score, scorer_name, None

        bar_traces, bar_x_axis, bar_y_axis = drift_score_bar_traces(score, bar_max=1)
        dist_traces, dist_x_axis, dist_y_axis = feature_distribution_traces(
                                                                train_dist, test_dist, value_name, is_categorical=True,
                                                                max_num_categories=max_num_categories_for_display,
                                                                show_categories_by=show_categories_by,
                                                                dataset_names=dataset_names)
    else:
        # Should never reach here
        raise DeepchecksValueError(f'Unsupported column type for drift: {column_type}')

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, shared_yaxes=False, shared_xaxes=False,
                        row_heights=[0.1, 0.9],
                        subplot_titles=[f'Drift Score ({scorer_name})', 'Distribution Plot'])

    fig.add_traces(bar_traces, rows=1, cols=1)
    fig.update_xaxes(bar_x_axis, row=1, col=1)
    fig.update_yaxes(bar_y_axis, row=1, col=1)
    fig.add_traces(dist_traces, rows=2, cols=1)
    fig.update_xaxes(dist_x_axis, row=2, col=1)
    fig.update_yaxes(dist_y_axis, row=2, col=1)

    fig.update_layout(
        legend=dict(
            title='Legend',
            yanchor='top',
            y=0.6),
        height=400,
        title=dict(text=plot_title or value_name, x=0.5, xanchor='center'),
        bargroupgap=0)

    return score, scorer_name, fig


def get_drift_plot_sidenote(max_num_categories_for_display: int, show_categories_by: str) -> str:
    """
    Return a sidenote for the drift score plots regarding the number of categories shown in discrete distributions.

    Parameters
    ----------
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:
        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.
    Returns
    -------
    str
        sidenote for the drift score plots regarding the number of categories shown in discrete distributions.
    """
    param_to_print_dict = {
        'train_largest': 'largest categories (by train)',
        'test_largest': 'largest categories (by test)',
        'largest_difference': 'categories with largest difference between train and test'
    }
    return f'For discrete distribution plots, ' \
           f'showing the top {max_num_categories_for_display} {param_to_print_dict[show_categories_by]}.'


def drift_condition(max_allowed_categorical_score: float,
                    max_allowed_numeric_score: float,
                    subject_single: str,
                    subject_multi: str,
                    allowed_num_subjects_exceeding_threshold: int = 0):
    """Create a condition function to be used in drift check's conditions.

    Parameters
    ----------
    max_allowed_categorical_score: float
        Max value allowed for categorical drift
    max_allowed_numeric_score: float
        Max value allowed for numerical drift
    subject_single: str
        String that represents the subject being tested as single (feature, column, property)
    subject_multi: str
        String that represents the subject being tested as multiple (features, columns, properties)
    allowed_num_subjects_exceeding_threshold: int, default: 0
        Determines the number of properties with drift score above threshold needed to fail the condition.
    """

    def condition(result: dict):
        cat_method, num_method = get_drift_method(result)
        cat_drift_props = {prop: d['Drift score'] for prop, d in result.items()
                           if d['Method'] in SUPPORTED_CATEGORICAL_METHODS}
        not_passing_categorical_props = {props: format_number(d) for props, d in cat_drift_props.items()
                                         if d >= max_allowed_categorical_score}
        num_drift_props = {prop: d['Drift score'] for prop, d in result.items()
                           if d['Method'] in SUPPORTED_NUMERIC_METHODS}
        not_passing_numeric_props = {prop: format_number(d) for prop, d in num_drift_props.items()
                                     if d >= max_allowed_numeric_score}

        num_failed = len(not_passing_categorical_props) + len(not_passing_numeric_props)
        if num_failed > allowed_num_subjects_exceeding_threshold:
            details = f'Failed for {num_failed} out of {len(result)} {subject_multi}.'
            if not_passing_categorical_props:
                details += f'\nFound {len(not_passing_categorical_props)} categorical {subject_multi} with ' \
                           f'{cat_method} above threshold: {not_passing_categorical_props}'
            if not_passing_numeric_props:
                details += f'\nFound {len(not_passing_numeric_props)} numeric {subject_multi} with {num_method} above' \
                           f' threshold: {not_passing_numeric_props}'
            return ConditionResult(ConditionCategory.FAIL, details)
        else:
            details = f'Passed for {len(result) - num_failed} {subject_multi} out of {len(result)} {subject_multi}.'
            if cat_drift_props:
                prop, score = get_dict_entry_by_value(cat_drift_props)
                details += f'\nFound {subject_single} "{prop}" has the highest categorical drift score: ' \
                           f'{format_number(score)}'
            if num_drift_props:
                prop, score = get_dict_entry_by_value(num_drift_props)
                details += f'\nFound {subject_single} "{prop}" has the highest numerical drift score: ' \
                           f'{format_number(score)}'
            return ConditionResult(ConditionCategory.PASS, details)

    return condition
