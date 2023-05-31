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
"""Module contains Prediction Drift check."""

import typing as t
from numbers import Number

import numpy as np

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.reduce_classes import ReduceMixin
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.abstracts.prediction_drift import PredictionDriftAbstract

__all__ = ['PredictionDrift']


class PredictionDrift(PredictionDriftAbstract, TrainTestCheck, ReduceMixin):
    """
    Calculate prediction drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the prediction in the test dataset, by comparing its distribution to the train
    dataset.
    For classification tasks, by default the drift score will be computed on the predicted probability of the positive
    (1) class for binary classification tasks, and on the predicted class itself for multiclass tasks. This behavior can
    be controlled using the `drift_mode` parameter.

    For numerical columns, we use the Kolmogorov-Smirnov statistic.
    See https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    We also support Earth Mover's Distance (EMD).
    See https://en.wikipedia.org/wiki/Wasserstein_metric

    For categorical distributions, we use the Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    We also support Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.

    For categorical predictions, it is recommended to use Cramer's V, unless your variable includes categories with a
    small number of samples (common practice is categories with less than 5 samples).
    However, in cases of a variable with many categories with few samples, it is still recommended to use Cramer's V.

    **Note:** In case of highly imbalanced classes, it is recommended to use Cramer's V, together with setting
    the ``balance_classes`` parameter to ``True``. This also requires setting the ``drift_mode`` parameter to
    ``auto`` (default) or ``'prediction'``.


    Parameters
    ----------
    drift_mode: str, default: 'auto'
        For classification task, controls whether to compute drift on the predicted probabilities or the predicted
        classes. For regression task this parameter may be ignored.
        If set to 'auto', compute drift on the predicted class if the task is multiclass, and on
        the predicted probability of the positive class if binary. Set to 'proba' to force drift on the predicted
        probabilities, and 'prediction' to force drift on the predicted classes. If set to 'proba', on a multiclass
        task, drift would be calculated on each class independently.
        If balance_classes=True, then 'auto' will calculate drift on the predicted class even if the label is binary
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category. Ignored if balance_classes=True.
    max_num_categories_for_drift: int, default: None
        Only relevant if drift is calculated for classification predictions. Max number of allowed categories.
        If there are more, they are binned into an "Other" category.
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
    balance_classes: bool, default: False
        If True, all categories will have an equal weight in the Cramer's V score. This is useful when the categorical
        variable is highly imbalanced, and we want to be alerted on changes in proportion to the category size,
        and not only to the entire dataset. Must have categorical_drift_method = "cramers_v" and
        drift_mode = "auto" or "prediction".
        If True, the variable frequency plot will be created with a log scale in the y-axis.
    ignore_na: bool, default True
        For categorical columns only. If True, ignores nones for categorical drift. If False, considers none as a
        separate category. For numerical columns we always ignore nones.
    aggregation_method: t.Optional[str], default: "max"
        Argument for the reduce_output functionality, decides how to aggregate the drift scores of different classes
        (for classification tasks) into a single score, when drift is computed on the class probabilities. Possible
        values are:
        'max': Maximum of all the class drift scores.
        'weighted': Weighted mean based on the class sizes in the train data set.
        'mean': Mean of all drift scores.
        None: No averaging. Return a dict with a drift score for each class.
    max_classes_to_display: int, default: 3
        Max number of classes to show in the display when drift is computed on the class probabilities for
        classification tasks.
    min_samples : int , default: 10
        Minimum number of samples required to calculate the drift score. If there are not enough samples for either
        train or test, the check will raise a ``NotEnoughSamplesError`` exception.
    n_samples : int , default: 100_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
            self,
            drift_mode: str = 'auto',
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = None,
            min_category_size_ratio: float = 0.01,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            numerical_drift_method: str = 'KS',
            categorical_drift_method: str = 'cramers_v',
            balance_classes: bool = False,
            ignore_na: bool = True,
            aggregation_method: t.Optional[str] = 'max',
            max_classes_to_display: int = 3,
            min_samples: t.Optional[int] = 10,
            n_samples: int = 100_000,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        if not isinstance(drift_mode, str):
            raise DeepchecksValueError('drift_mode must be a string')
        self.drift_mode = drift_mode.lower()
        if self.drift_mode not in ('auto', 'proba', 'prediction'):
            raise DeepchecksValueError('drift_mode must be one of "auto", "proba", "prediction"')
        self.margin_quantile_filter = margin_quantile_filter
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.min_category_size_ratio = min_category_size_ratio
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.numerical_drift_method = numerical_drift_method
        self.categorical_drift_method = categorical_drift_method
        self.balance_classes = balance_classes
        if self.balance_classes is True and self.drift_mode == 'proba':
            raise DeepchecksValueError('balance_classes=True is not supported for drift_mode=\'proba\'. '
                                       'Change drift_mode to \'prediction\' or \'auto\' in order to use this parameter')
        self.ignore_na = ignore_na
        self.max_classes_to_display = max_classes_to_display
        self.aggregation_method = aggregation_method
        self.min_samples = min_samples
        self.n_samples = n_samples
        self.random_state = random_state
        if self.aggregation_method not in ('weighted', 'mean', 'none', None, 'max'):
            raise DeepchecksValueError('aggregation_method must be one of "weighted", "mean", "max", None')

    def run_logic(self, context: Context) -> CheckResult:
        """Calculate drift for all columns.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """
        if (self.drift_mode == 'proba') and (context.task_type == TaskType.REGRESSION):
            raise DeepchecksValueError('probability_drift="proba" is not supported for regression tasks')

        train_dataset = context.train.sample(self.n_samples, random_state=self.random_state)
        test_dataset = context.test.sample(self.n_samples, random_state=self.random_state)
        model = context.model

        # Flag for computing drift on the probabilities rather than the predicted labels
        proba_drift = \
            ((context.task_type == TaskType.BINARY and self.drift_mode == 'auto' and hasattr(model, 'predict_proba'))
             or (self.drift_mode == 'proba')) \
            and not (self.balance_classes is True and self.drift_mode == 'auto')

        if proba_drift:
            train_pred = np.array(model.predict_proba(train_dataset.features_columns))
            test_pred = np.array(model.predict_proba(test_dataset.features_columns))
        else:
            train_pred = np.array(model.predict(train_dataset.features_columns)).reshape((-1, 1))
            test_pred = np.array(model.predict(test_dataset.features_columns)).reshape((-1, 1))

        return self._prediction_drift(train_pred, test_pred, context.model_classes, context.with_display, proba_drift,
                                      (context.task_type != TaskType.REGRESSION) and (not proba_drift))

    def reduce_output(self, check_result: CheckResult) -> t.Dict[str, float]:
        """Return prediction drift score."""
        if isinstance(check_result.value['Drift score'], Number):
            return {'Prediction Drift Score': check_result.value['Drift score']}

        drift_values = list(check_result.value['Drift score'].values())
        if self.aggregation_method is None or self.aggregation_method == 'none':
            return {f'Drift Score class {k}': v for k, v in check_result.value['Drift score'].items()}
        elif self.aggregation_method == 'mean':
            return {'Mean Drift Score': np.mean(drift_values)}
        elif self.aggregation_method == 'max':
            return {'Max Drift Score': np.max(drift_values)}
        elif self.aggregation_method == 'weighted':
            class_weight = np.array([check_result.value['Samples per class'][class_name] for class_name in
                                     check_result.value['Drift score'].keys()])
            class_weight = class_weight / np.sum(class_weight)
            return {'Weighted Drift Score': np.sum(np.array(drift_values) * class_weight)}

    def greater_is_better(self):
        """Return True if the check reduce_output is better when it is greater."""
        return False
