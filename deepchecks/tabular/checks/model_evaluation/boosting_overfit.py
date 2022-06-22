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
"""Boosting overfit check module."""
from copy import deepcopy
from typing import Callable, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError, ModelValidationError
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.utils.model import get_model_of_pipeline
from deepchecks.utils.strings import format_percent

__all__ = ['BoostingOverfit']


class PartialBoostingModel:
    """Wrapper for boosting models which limits the number of estimators being used in the prediction."""

    _UNSUPORTED_MODEL_ERROR = (
        'Check is relevant for Boosting models of type '
        '{supported_models}, but received model of type {model_type}'
    )

    _SUPPORTED_CLASSIFICATION_MODELS = (
        'AdaBoostClassifier',
        'GradientBoostingClassifier',
        'LGBMClassifier',
        'XGBClassifier',
        'CatBoostClassifier'
    )

    _SUPPORTED_REGRESSION_MODELS = (
        'AdaBoostRegressor',
        'GradientBoostingRegressor',
        'LGBMRegressor',
        'XGBRegressor',
        'CatBoostRegressor'
    )

    _SUPPORTED_MODELS = _SUPPORTED_CLASSIFICATION_MODELS + _SUPPORTED_REGRESSION_MODELS

    def __init__(self, model, step):
        """Construct wrapper for model with `predict` and `predict_proba` methods.

        Parameters
        ----------
        model
            boosting model to wrap.
        step
            Number of iterations/estimators to limit the model on predictions.
        """
        self.model_class = get_model_of_pipeline(model).__class__.__name__
        self.step = step
        if self.model_class in ['AdaBoostClassifier', 'GradientBoostingClassifier', 'AdaBoostRegressor',
                                'GradientBoostingRegressor']:
            self.model = deepcopy(model)
            if isinstance(model, Pipeline):
                internal_estimator = get_model_of_pipeline(self.model)
                internal_estimator.estimators_ = internal_estimator.estimators_[:self.step]
            else:
                self.model.estimators_ = self.model.estimators_[:self.step]
        else:
            self.model = model

    def predict_proba(self, x):
        if self.model_class in ['AdaBoostClassifier', 'GradientBoostingClassifier']:
            return self.model.predict_proba(x)
        elif self.model_class == 'LGBMClassifier':
            return self.model.predict_proba(x, num_iteration=self.step)
        elif self.model_class == 'XGBClassifier':
            return self.model.predict_proba(x, iteration_range=(0, self.step))
        elif self.model_class == 'CatBoostClassifier':
            return self.model.predict_proba(x, ntree_end=self.step)
        else:
            raise ModelValidationError(self._UNSUPORTED_MODEL_ERROR.format(
                supported_models=self._SUPPORTED_CLASSIFICATION_MODELS,
                model_type=self.model_class
            ))

    def predict(self, x):
        if self.model_class in ['AdaBoostClassifier', 'GradientBoostingClassifier', 'AdaBoostRegressor',
                                'GradientBoostingRegressor']:
            return self.model.predict(x)
        elif self.model_class in ['LGBMClassifier', 'LGBMRegressor']:
            return self.model.predict(x, num_iteration=self.step)
        elif self.model_class in ['XGBClassifier', 'XGBRegressor']:
            return self.model.predict(x, iteration_range=(0, self.step))
        elif self.model_class in ['CatBoostClassifier', 'CatBoostRegressor']:
            return self.model.predict(x, ntree_end=self.step)
        else:
            raise ModelValidationError(self._UNSUPORTED_MODEL_ERROR.format(
                supported_models=self._SUPPORTED_MODELS,
                model_type=self.model_class
            ))

    @classmethod
    def n_estimators(cls, model):
        model = get_model_of_pipeline(model)
        model_class = model.__class__.__name__
        if model_class in ['AdaBoostClassifier', 'GradientBoostingClassifier', 'AdaBoostRegressor',
                           'GradientBoostingRegressor']:
            return len(model.estimators_)
        elif model_class in ['LGBMClassifier', 'LGBMRegressor']:
            return model.n_estimators
        elif model_class in ['XGBClassifier', 'XGBRegressor']:
            return model.n_estimators
        elif model_class in ['CatBoostClassifier', 'CatBoostRegressor']:
            return model.tree_count_
        else:
            raise ModelValidationError(cls._UNSUPORTED_MODEL_ERROR.format(
                supported_models=cls._SUPPORTED_MODELS,
                model_type=model_class
            ))


class BoostingOverfit(TrainTestCheck):
    """Check for overfit caused by using too many iterations in a gradient boosted model.

    The check runs a pred-defined number of steps, and in each step it limits the boosting model to use up to X
    estimators (number of estimators is monotonic increasing). It plots the given score calculated for each step for
    both the train dataset and the test dataset.

    Parameters
    ----------
    scorer : Union[Callable, str] , default: None
        Scorer used to verify the model, either function or sklearn scorer name.
    scorer_name : str , default: None
        Name to be displayed in the plot on y-axis. must be used together with 'scorer'
    num_steps : int , default: 20
        Number of splits of the model iterations to check.
    """

    def __init__(
        self,
        alternative_scorer: Tuple[str, Union[str, Callable]] = None,
        num_steps: int = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.user_scorer = dict([alternative_scorer]) if alternative_scorer else None
        self.num_steps = num_steps

        if not isinstance(self.num_steps, int) or self.num_steps < 2:
            raise DeepchecksValueError('num_steps must be an integer larger than 1')

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            The score value on the test dataset.
        """
        train_dataset = context.train
        test_dataset = context.test
        model = context.model

        # Get default scorer
        scorer = context.get_single_scorer(self.user_scorer)

        # Get number of estimators on model
        num_estimators = PartialBoostingModel.n_estimators(model)
        estimator_steps = _calculate_steps(self.num_steps, num_estimators)

        train_scores = []
        test_scores = []
        for step in estimator_steps:
            train_scores.append(_partial_score(scorer, train_dataset, model, step))
            test_scores.append(_partial_score(scorer, test_dataset, model, step))

        result = {'test': test_scores, 'train': train_scores}

        if context.with_display:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=estimator_steps, y=np.array(train_scores),
                                     mode='lines+markers',
                                     name='Training score'))
            fig.add_trace(go.Scatter(x=estimator_steps, y=np.array(test_scores),
                                     mode='lines+markers',
                                     name='Test score'))
            fig.update_layout(
                title_text=f'{scorer.name} score compared to number of boosting iteration',
                height=500
            )
            fig.update_xaxes(title='Number of boosting iterations')
            fig.update_yaxes(title=scorer.name)

            display_text = f"""<span>
                The check limits the boosting model to using up to N estimators each time, and plotting the
                {scorer.name} calculated for each subset of estimators for both the train dataset and the test dataset.
            </span>"""
            display = [display_text, fig]
        else:
            display = None

        return CheckResult(result, display=display, header='Boosting Overfit')

    def add_condition_test_score_percent_decline_less_than(self, threshold: float = 0.05):
        """Add condition.

        Percent of decline between the maximal score achieved in any boosting iteration and the score achieved in the
        last iteration ("regular" model score) is not above given threshold.

        Parameters
        ----------
        threshold : float , default: 0.05
            Maximum percentage decline allowed (value 0 and above)
        """
        def condition(result: dict):
            max_score = max(result['test'])
            last_score = result['test'][-1]
            pct_diff = (max_score - last_score) / abs(max_score)
            details = f'Found score decline of {format_percent(-pct_diff)}'
            category = ConditionCategory.PASS if pct_diff < threshold else ConditionCategory.FAIL
            return ConditionResult(category, details)

        name = f'Test score over iterations is less than {format_percent(threshold)} from the best score'
        return self.add_condition(name, condition)


def _partial_score(scorer, dataset, model, step):
    partial_model = PartialBoostingModel(model, step)
    return scorer(partial_model, dataset)


def _calculate_steps(num_steps, num_estimators):
    """Calculate steps (integers between 1 to num_estimators) to work on."""
    if num_steps >= num_estimators:
        return list(range(1, num_estimators + 1))
    if num_steps <= 5:
        steps_percents = np.linspace(0, 1.0, num_steps + 1)[1:]
        steps_numbers = np.ceil(steps_percents * num_estimators)
        steps_set = {int(s) for s in steps_numbers}
    else:
        steps_percents = np.linspace(5 / num_estimators, 1.0, num_steps - 4)[1:]
        steps_numbers = np.ceil(steps_percents * num_estimators)
        steps_set = {int(s) for s in steps_numbers}
        # We want to forcefully take the first 5 estimators, since they have the largest affect on the model performance
        steps_set.update({1, 2, 3, 4, 5})

    return sorted(steps_set)
