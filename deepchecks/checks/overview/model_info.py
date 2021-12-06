# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Module contains model_info check."""
import pandas as pd
from sklearn.base import BaseEstimator

from deepchecks import ModelOnlyBaseCheck, CheckResult
from deepchecks.utils.validation import model_type_validation


__all__ = ['ModelInfo']


class ModelInfo(ModelOnlyBaseCheck):
    """Summarize given model parameters."""

    def run(self, model: BaseEstimator) -> CheckResult:
        """Run check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is dictionary in format {type: <model_type>, params: <model_params_dict>}
        """
        return self._model_info(model)

    def _model_info(self, model: BaseEstimator):
        model_type_validation(model)
        model_type = type(model).__name__
        model_params = model.get_params()
        default_params = type(model)().get_params()

        # Create dataframe to show
        model_param_df = pd.DataFrame(model_params.items(), columns=['Parameter', 'Value'])
        model_param_df['Default'] = model_param_df['Parameter'].map(lambda x: default_params[x])

        def highlight_not_default(data):
            n = len(data)
            param = data[0]
            value = data[1]
            if value != default_params[param]:
                return n * ['background-color: lightblue']
            else:
                return n * ['']

        model_param_df = model_param_df.style.apply(highlight_not_default, axis=1).hide_index()

        value = {'type': model_type, 'params': model_params}
        footnote = '<p style="font-size:0.7em"><i>Colored rows are parameters with non-default values</i></p>'
        display = [f'Model Type: {model_type}', model_param_df, footnote]

        return CheckResult(value, check=self.__class__, header='Model Info', display=display)
