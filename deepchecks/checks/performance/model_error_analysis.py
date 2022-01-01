# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module of segment performance check."""
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult, ConditionCategory
from deepchecks.errors import DeepchecksProcessError
from deepchecks.utils.features import calculate_feature_importance
from deepchecks.utils.metrics import task_type_check, ModelType, initialize_multi_scorers, get_scorers_list, \
    get_scorer_single
from deepchecks.utils.plot import colors
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.utils.validation import validate_model

__all__ = ['ModelErrorAnalysis']


class ModelErrorAnalysis(TrainTestBaseCheck):
    """Find features that best split the data into segments of high and low model error.

    The check trains a regression model to predict the error of the user's model. Then, the features scoring the highest
    feature importance for the error regression model are selected and the distribution of the error vs the feature
    values is plotted. The check results are shown only if the error regression model manages to predict the error
    well enough.

    Args:
        max_features (int): maximal number of features to show. (default: 3)
        min_feature_contribution (float): minimum feature importance of a feature to the error regression model
            in order to show the feature. (default: 0.15)
        min_error_model_score (float): minimum r^2 score of the error regression model for displaying the
            check. (default: 0.5)
        min_segment_size (float): minimal fraction of data that can comprise a weak segment. (default: 0.1)
        alternative_scorer (Dict[str, Callable], default None):
            An optional dictionary of scorer name to scorer function. Only a single entry is allowed in this check.
            If none given, using default scorer
        n_samples (int): number of samples to use for this check. (default: 50000)
        n_display_samples (int): number of samples to display in scatter plot. (default: 5000)
        random_seed (int): random seed for all check internals. (default: 42)
    """

    def __init__(
            self,
            max_features: int = 3,
            min_feature_contribution: float = 0.15,
            min_error_model_score: float = 0.5,
            min_segment_size: float = 0.05,
            alternative_scorer: Dict[str, Callable] = None,
            n_samples: int = 50_000,
            n_display_samples: int = 5_000,
            random_seed: int = 42
    ):
        super().__init__()
        self.max_features = max_features
        self.min_feature_contribution = min_feature_contribution
        self.min_error_model_score = min_error_model_score
        self.min_segment_size = min_segment_size
        if (alternative_scorer is not None) and (len(alternative_scorer) > 1):
            raise DeepchecksProcessError(
                'Only one alternative scorer is allowed.'
            )
        self.alternative_scorer = initialize_multi_scorers(alternative_scorer)
        self.n_samples = n_samples
        self.n_display_samples = n_display_samples
        self.random_state = random_seed

        self._scorer_name = None

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label.
            test_dataset (Dataset): The test dataset object. Must contain a label.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        """
        # Validations
        Dataset.validate_dataset(train_dataset)
        Dataset.validate_dataset(test_dataset)
        train_dataset.validate_label()
        test_dataset.validate_label()
        train_dataset.validate_shared_label(test_dataset)
        train_dataset.validate_shared_features(test_dataset)
        train_dataset.validate_shared_categorical_features(test_dataset)
        validate_model(train_dataset, model)

        task_type = task_type_check(model, train_dataset)

        # If user defined scorers used them, else use a single scorer
        if self.alternative_scorer:
            scorer = get_scorers_list(model, train_dataset, self.alternative_scorer, multiclass_avg=True)[0]
        else:
            scorer = get_scorer_single(model, train_dataset, multiclass_avg=True)

        cat_features = train_dataset.cat_features

        train_dataset = train_dataset.sample(self.n_samples, random_state=self.random_state)
        test_dataset = test_dataset.sample(self.n_samples, random_state=self.random_state)

        # Create scoring function, used to calculate the per sample model error
        if task_type == ModelType.REGRESSION:
            def scoring_func(dataset):
                return per_sample_mse(dataset.label_col, model.predict(dataset.features_columns))
        else:
            def scoring_func(dataset):
                return per_sample_binary_cross_entropy(dataset.label_col,
                                                       model.predict_proba(dataset.features_columns))

        train_scores = scoring_func(train_dataset)
        test_scores = scoring_func(test_dataset)

        # Create and fit model to predict the per sample error
        error_model = create_error_model(train_dataset, random_state=self.random_state)
        error_model.fit(train_dataset.features_columns, y=train_scores)

        # Check if fitted model is good enough
        error_model_predicted = error_model.predict(test_dataset.features_columns)
        error_model_score = r2_score(test_scores, error_model_predicted)

        # This check should be ignored if no information gained from the error model (low r2_score)
        if error_model_score < self.min_error_model_score:
            raise DeepchecksProcessError(f'Unable to train meaningful error model '
                                         f'(r^2 score: {format_number(error_model_score)})')

        error_fi = calculate_feature_importance(error_model, test_dataset,
                                                permutation_kwargs={'random_state': self.random_state})
        error_fi.sort_values(ascending=False, inplace=True)

        n_samples_display = min(self.n_display_samples, len(test_dataset))
        display_data = test_dataset.data
        error_col_name = 'Deepchecks model error'
        display_error = pd.Series(error_model_predicted, name=error_col_name, index=test_dataset.data.index)

        display = []
        value = {}
        self._scorer_name = scorer.name
        weak_color = '#d74949'
        ok_color = colors['Test']

        for feature in error_fi.keys()[:self.max_features]:
            if error_fi[feature] < self.min_feature_contribution:
                break

            data = pd.concat([display_data[feature], display_error], axis=1)

            # Violin plot for categorical features, scatter plot for numerical features
            if feature in cat_features:
                # find categories with the weakest performance
                error_per_segment_ser = data.groupby(feature).agg(['mean', 'count']
                                                                  )[error_col_name].sort_values('mean', ascending=False)
                cum_sum_ratio = error_per_segment_ser['count'].cumsum() / error_per_segment_ser['count'].sum()

                # Partition data into two groups - weak and ok:

                in_segment_indicis = cum_sum_ratio <= self.min_segment_size
                weak_categories = error_per_segment_ser.index[in_segment_indicis]
                ok_categories = error_per_segment_ser.index[~in_segment_indicis]

                # Calculate score for each group and assign label and color
                ok_name_feature, segment1_details = get_segment_details(model, scorer, test_dataset, data,
                                                                        data[feature].isin(ok_categories))

                color_map = {ok_name_feature: ok_color}

                if len(weak_categories) >= 1:
                    weak_name_feature, segment2_details = get_segment_details(model, scorer, test_dataset, data,
                                                                              data[feature].isin(weak_categories))

                    color_map[weak_name_feature] = weak_color
                else:
                    weak_name_feature = None

                replace_dict = {x: weak_name_feature if x in weak_categories else ok_name_feature for x in
                                error_per_segment_ser.index}
                color_col = data[feature].replace(replace_dict)

                # Display
                display.append(px.violin(
                    data, y=error_col_name, x=feature, title=f'Segmentation of error by {feature}', box=False,
                    labels={error_col_name: 'model error'}, color=color_col,
                    color_discrete_map=color_map
                ))
            else:
                data = data.sample(n_samples_display, random_state=self.random_state)

                # Train tree to partition segments according to the model error
                tree_partitioner = DecisionTreeRegressor(max_depth=1,
                                                         min_samples_leaf=self.min_segment_size + np.finfo(float).eps,
                                                         random_state=self.random_state).fit(
                    data[[feature]], data[error_col_name])
                if len(tree_partitioner.tree_.threshold) > 1:
                    threshold = tree_partitioner.tree_.threshold[0]
                    color_col = data[feature].ge(threshold)

                    segment1_text, segment1_details = get_segment_details(model, scorer, test_dataset, data,
                                                                          color_col)
                    segment2_text, segment2_details = get_segment_details(model, scorer, test_dataset, data,
                                                                          ~color_col)
                    color_col = color_col.replace([True, False], [segment1_text, segment2_text])
                    if segment1_details['score'] >= segment2_details['score']:
                        color_map = {segment1_text: ok_color, segment2_text: weak_color}
                    else:
                        color_map = {segment1_text: weak_color, segment2_text: ok_color}
                else:
                    color_col = data[error_col_name]
                    color_map = None
                display.append(px.scatter(data, x=feature, y=error_col_name, color=color_col,
                                          title=f'Segmentation of error by {feature}',
                                          labels={error_col_name: 'model error'},
                                          color_discrete_map=color_map))

            value[feature] = {'segment1': segment1_details or None, 'segment2': segment2_details or None}

            display[-1].update_layout(width=1200, height=400)

        headnote = """<span>
            The following graphs show the distribution of error for top features that are most useful for distinguishing
            high error samples from low error samples.
        </span>"""
        display = [headnote] + display if display else None

        return CheckResult(value, display=display)

    def add_condition_segments_ratio_performance_change_not_greater_than(self, max_ratio_change: float = 0.05):
        """Add condition - require that the difference of performance between the segments does not exceed a ratio.

        Args:
            max_ratio_change (float): maximal ratio of change between the two segments' performance.
        """

        def condition(result: Dict) -> ConditionResult:
            fails = []
            for feature in result.keys():
                # If only one segment identified
                if len(result[feature]) < 2:
                    continue
                performance_diff = \
                    abs(result[feature]['segment1']['score'] - result[feature]['segment2']['score']) / \
                    abs(max(result[feature]['segment1']['score'], result[feature]['segment2']['score']))
                if performance_diff > max_ratio_change:
                    fails.append(feature)

            if fails:
                msg = f'Segmentation of error by the features: {", ".join(sorted(fails))} resulted in percent change' \
                      f' in {self._scorer_name} larger than {format_percent(max_ratio_change)}.'
                return ConditionResult(False, msg, category=ConditionCategory.WARN)
            else:
                return ConditionResult(True, category=ConditionCategory.WARN)

        return self.add_condition(f'The percent change between the performance of detected segments must'
                                  f' not exceed {format_percent(max_ratio_change)}', condition)


def get_segment_details(model, scorer, dataset: Dataset, data: pd.DataFrame,
                        segment_condition_col: pd.Series) -> Tuple[str, Dict[str, float]]:
    """Return a string with details about the data segment."""
    performance = scorer(
        model,
        dataset.copy(dataset.data.loc[data.index][segment_condition_col]))
    n_samples = data[segment_condition_col].shape[0]
    segment_label = \
        f'{scorer.name}: {format_number(performance)}, ' \
        f'Samples: {n_samples} ({format_percent(n_samples / len(data))})'

    segment_details = {'score': performance, 'n_samples': n_samples, 'frac_samples': n_samples / len(data)}

    return segment_label, segment_details


def per_sample_binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true)
    return - (np.tile(y_true.reshape((-1, 1)), (1, y_pred.shape[1])) *
              np.log(y_pred + np.finfo(float).eps)).sum(axis=1)


def per_sample_mse(y_true, y_pred):
    return (y_true - y_pred) ** 2


def create_error_model(dataset: Dataset, random_state=42):
    cat_features = dataset.cat_features
    numeric_features = [num_feature for num_feature in dataset.features if num_feature not in cat_features]

    numeric_transformer = SimpleImputer()
    categorical_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', TargetEncoder())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, cat_features),
        ]
    )

    return Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', RandomForestRegressor(max_depth=4, n_jobs=-1, random_state=random_state))
    ])
