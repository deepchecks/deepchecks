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
"""Module contains model_info check."""
import pandas as pd
from sklearn.base import BaseEstimator

from deepchecks import ModelOnlyBaseCheck, CheckResult
from deepchecks.utils.model import get_model_of_pipeline
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
        model_type_validation(model)
        estimator = get_model_of_pipeline(model)
        model_type = type(estimator).__name__
        model_params = estimator.get_params()
        default_params = type(estimator)().get_params()

        # Create dataframe to show
        model_param_df = pd.DataFrame(model_params.items(), columns=['Parameter', 'Value'])
        model_param_df['Default'] = model_param_df['Parameter'].map(lambda x: default_params.get(x, ''))

        def highlight_not_default(data):
            n = len(data)
            if data['Value'] != data['Default']:
                return n * ['background-color: lightblue']
            else:
                return n * ['']

        model_param_df = model_param_df.style.apply(highlight_not_default, axis=1).hide_index()

        value = {'type': model_type, 'params': model_params}
        footnote = '<p style="font-size:0.7em"><i>Colored rows are parameters with non-default values</i></p>'
        display = [f'Model Type: {model_type}', model_param_df, footnote]

        return CheckResult(value, header='Model Info', display=display)
