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
"""Module contains Property Drift check."""
import textwrap
import typing as t

from deepchecks.core import CheckResult
from deepchecks.core.errors import NotEnoughSamplesError
from deepchecks.nlp.base_checks import TrainTestCheck
from deepchecks.nlp.context import Context
from deepchecks.nlp.text_data import TextData
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.distribution.drift import calc_drift_and_plot, drift_condition, get_drift_plot_sidenote
from deepchecks.utils.typing import Hashable

__all__ = ['PropertyDrift']


# TODO:
# refactor, separate general drift logic into separate class/module and use it with drift checks
@docstrings
class PropertyDrift(TrainTestCheck):
    """
    Calculate drift between train dataset and test dataset per feature, using statistical measures.

    Check calculates a drift score for each column in test dataset, by comparing its distribution to the train
    dataset.

    For numerical columns, we use the Kolmogorov-Smirnov statistic.
    See https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    We also support Earth Mover's Distance (EMD).
    See https://en.wikipedia.org/wiki/Wasserstein_metric

    For categorical distributions, we use the Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    We also support Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.

    For categorical variables, it is recommended to use Cramer's V, unless your variable includes categories with a
    small number of samples (common practice is categories with less than 5 samples).
    However, in cases of a variable with many categories with few samples, it is still recommended to use Cramer's V.

    Parameters
    ----------
    properties : Union[Hashable, List[Hashable]] , default: None
        Properties to check, if none is given, checks all
        properties except ignored ones.
    ignore_properties : Union[Hashable, List[Hashable]] , default: None
        Properties to ignore, if none is given, checks based on
        properties variable.
    n_top_properties : int , default 5
        amount of properties to show ordered by drift score
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category.
    max_num_categories_for_drift: Optional[int], default: None
        Only for categorical features. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. This limit applies for both drift calculation and distribution plots.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:
        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.
    numerical_drift_method: str, default: "KS"
        decides which method to use on numerical variables. Possible values are:
        "EMD" for Earth Mover's Distance (EMD), "KS" for Kolmogorov-Smirnov (KS).
    categorical_drift_method: str, default: "cramers_v"
        decides which method to use on categorical variables. Possible values are:
        "cramers_v" for Cramer's V, "PSI" for Population Stability Index (PSI).
    ignore_na: bool, default True
        For categorical columns only. If True, ignores nones for categorical drift. If False, considers none as a
        separate category. For numerical columns we always ignore nones.
    aggregation_method: Optional[str], default: 'l3_weighted'
        {feature_aggregation_method_argument:2*indent}
    min_samples : int , default: 10
        Minimum number of samples required to calculate the drift score. If there are not enough samples for either
        train or test, the check will return None for that feature. If there are not enough samples for all properties,
        the check will raise a ``NotEnoughSamplesError`` exception.
    n_samples : int , default: 100_000
        Number of samples to use for drift computation and plot.
    random_state : int , default: 42
        Random seed for sampling.
    """

    def __init__(
        self,
        properties: t.Union[t.Hashable, t.List[Hashable], None] = None,
        ignore_properties: t.Union[Hashable, t.List[Hashable], None] = None,
        n_top_properties: int = 5,
        margin_quantile_filter: float = 0.025,
        max_num_categories_for_drift: t.Optional[int] = None,
        min_category_size_ratio: float = 0.01,
        max_num_categories_for_display: int = 10,
        show_categories_by: str = 'largest_difference',
        numerical_drift_method: str = 'KS',
        categorical_drift_method: str = 'cramers_v',
        ignore_na: bool = True,
        aggregation_method: t.Optional[str] = 'l3_weighted',
        min_samples: int = 10,
        n_samples: int = 100_000,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.properties = properties
        self.ignore_properties = ignore_properties
        self.n_top_properties = n_top_properties
        self.margin_quantile_filter = margin_quantile_filter
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.min_category_size_ratio = min_category_size_ratio
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.numerical_drift_method = numerical_drift_method
        self.categorical_drift_method = categorical_drift_method
        self.ignore_na = ignore_na
        self.aggregation_method = aggregation_method
        self.min_samples = min_samples
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context) -> CheckResult:
        train = t.cast(TextData, context.train)
        test = t.cast(TextData, context.test)

        train_properties = select_from_dataframe(
            train.properties,
            columns=self.properties,
            ignore_columns=self.ignore_properties
        )
        test_properties = select_from_dataframe(
            test.properties,
            columns=self.properties,
            ignore_columns=self.ignore_properties
        )

        columns = set(train_properties.columns).intersection(set(test_properties.columns))
        cat_columns = set([*train.categorical_properties, *test.categorical_properties])
        results = {}
        plots = {}
        not_enough_samples = []

        for column_name in columns:
            score, method, display = calc_drift_and_plot(
                train_column=train_properties[column_name],
                test_column=test_properties[column_name],
                value_name=column_name,
                column_type=(
                    'categorical'
                    if column_name in cat_columns
                    else 'numerical'
                ),
                plot_title=f'Property {column_name}',
                margin_quantile_filter=self.margin_quantile_filter,
                max_num_categories_for_drift=self.max_num_categories_for_drift,
                min_category_size_ratio=self.min_category_size_ratio,
                max_num_categories_for_display=self.max_num_categories_for_display,
                show_categories_by=self.show_categories_by,
                numerical_drift_method=self.numerical_drift_method,
                categorical_drift_method=self.categorical_drift_method,
                ignore_na=self.ignore_na,
                min_samples=self.min_samples,
                with_display=context.with_display,
                dataset_names=(train.name or 'Train', test.name or 'Test')
            )

            if isinstance(score, str) and score == 'not_enough_samples':
                not_enough_samples.append(column_name)
                score = None
            else:
                plots[column_name] = display

            results[column_name] = {
                'Drift score': score,
                'Method': method,
            }

        if len(not_enough_samples) == len(results.keys()):
            raise NotEnoughSamplesError(
                f'Not enough samples to calculate drift score. Minimum {self.min_samples} samples required. '
                'Note that for numerical properties, None values do not count as samples.'
                'Use the \'min_samples\' parameter to change this requirement.'
            )

        if context.with_display:
            key = lambda column: results[column]['Drift score'] or 0
            sorted_properties = sorted(results.keys(), key=key, reverse=True)[:self.n_top_properties]

            headnote = [
                textwrap.dedent(f"""
                <span>
                The Drift score is a measure for the difference between two distributions, in this check - the test
                and train distributions.<br> The check shows the drift score and distributions for the properties,
                sorted by drift score and showing only the top {self.n_top_properties} properties.
                </span>
                """),
                get_drift_plot_sidenote(
                    self.max_num_categories_for_display,
                    self.show_categories_by
                )
            ]

            if not_enough_samples:
                headnote.append(
                    '<span>The following properties do not have enough samples to calculate drift '
                    f'score: {not_enough_samples}</span>'
                )

            displays = [
                *headnote,
                *(plots[p] for p in sorted_properties if results[p]['Drift score'] is not None)
            ]
        else:
            displays = None

        return CheckResult(
            value=results,
            display=displays,
            header='Properties Drift'
        )

    def add_condition_drift_score_less_than(
        self,
        max_allowed_categorical_score: float = 0.2,
        max_allowed_numeric_score: float = 0.2,
        allowed_num_features_exceeding_threshold: int = 0
    ):
        """
        Add condition - require drift score to be less than the threshold.

        The industry standard for PSI limit is above 0.2.
        There are no common industry standards for other drift methods, such as Cramer's V,
        Kolmogorov-Smirnov and Earth Mover's Distance.

        Parameters
        ----------
        max_allowed_categorical_score: float , default: 0.2
            The max threshold for the categorical variable drift score
        max_allowed_numeric_score: float ,  default: 0.2
            The max threshold for the numeric variable drift score
        allowed_num_features_exceeding_threshold: int , default: 0
            Determines the number of features with drift score above threshold needed to fail the condition.

        Returns
        -------
        ConditionResult
            False if more than allowed_num_features_exceeding_threshold drift scores are above threshold, True otherwise
        """
        condition = drift_condition(max_allowed_categorical_score, max_allowed_numeric_score, 'column', 'columns',
                                    allowed_num_features_exceeding_threshold)

        return self.add_condition(f'categorical drift score < {max_allowed_categorical_score} and '
                                  f'numerical drift score < {max_allowed_numeric_score}',
                                  condition)
