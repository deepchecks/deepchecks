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
"""Module contains model_info check."""
import warnings

import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.tabular import Context, ModelOnlyCheck
from deepchecks.utils.model import get_model_of_pipeline

__all__ = ['ModelInfo']


class ModelInfo(ModelOnlyCheck):
    """Summarize given model parameters."""

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is dictionary in format {type: <model_type>, params: <model_params_dict>}
        """
        model = context.model
        estimator = get_model_of_pipeline(model)
        model_type = type(estimator).__name__
        try:
            model_params = estimator.get_params()
            default_params = type(estimator)().get_params()
        except AttributeError:
            model_params = {}
            default_params = {}

        # Create dataframe to show
        model_param_df = pd.DataFrame(model_params.items(), columns=['Parameter', 'Value'])
        model_param_df['Default'] = model_param_df['Parameter'].map(lambda x: default_params.get(x, ''))

        def highlight_not_default(data):
            n = len(data)
            if data['Value'] != data['Default']:
                return n * ['background-color: lightblue']
            else:
                return n * ['']
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            model_param_df = model_param_df.style.apply(highlight_not_default, axis=1).hide_index()

        value = {'type': model_type, 'params': model_params}
        footnote = '<p style="font-size:0.7em"><i>Colored rows are parameters with non-default values</i></p>'
        display = [f'Model Type: {model_type}', model_param_df, footnote]

        return CheckResult(value, header='Model Info', display=display)
