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
import typing as t

from deepchecks.core import CheckResult
from deepchecks.nlp.base_checks import TrainTestCheck
from deepchecks.nlp.context import Context
from deepchecks.nlp.text_data import TextData
from deepchecks.nlp.utils.text_properties import TEXT_PROPERTIES_DESCRIPTION
from deepchecks.utils.abstracts.feature_drift import FeatureDriftAbstract
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import get_docs_link
from deepchecks.utils.typing import Hashable

__all__ = ['PropertyDrift']


class PropertyDrift(TrainTestCheck, FeatureDriftAbstract):
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
    min_samples : int , default: 100
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
        min_samples: int = 100,
        n_samples: int = 100_000,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.properties = properties
        self.ignore_properties = ignore_properties
        self.n_top_columns = n_top_properties
        self.margin_quantile_filter = margin_quantile_filter
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.min_category_size_ratio = min_category_size_ratio
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.numerical_drift_method = numerical_drift_method
        self.categorical_drift_method = categorical_drift_method
        self.ignore_na = ignore_na
        self.min_samples = min_samples
        self.n_samples = n_samples
        self.random_state = random_state
        self.sort_feature_by = 'drift score'

    def run_logic(self, context: Context) -> CheckResult:
        """Run check."""
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

        # Specialized plot titles for NLP Plots
        plot_titles = {}
        for column_name in context.common_datasets_properties:
            if column_name in TEXT_PROPERTIES_DESCRIPTION:
                plot_titles[column_name] = f'{column_name}<sup><a href="{get_docs_link()}nlp/usage_guides' \
                            '/nlp_properties.html#deepchecks-built-in-properties">&#x24D8;</a></sup><br>' \
                            f'<sup>{TEXT_PROPERTIES_DESCRIPTION[column_name]}</sup>'

        results, displays = self._calculate_feature_drift(
            drift_kind='nlp-properties',
            train=train_properties,
            test=test_properties,
            train_dataframe_name=train.name or 'Train',
            test_dataframe_name=test.name or 'Test',
            common_columns=context.common_datasets_properties,
            override_plot_titles=plot_titles,
            with_display=context.with_display
        )
        return CheckResult(
            value=results,
            display=displays,
            header='Property Drift'
        )
