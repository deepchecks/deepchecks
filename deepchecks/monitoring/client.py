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
"""Module for the monitoring client."""
from typing import Union, Optional, Dict

import requests


class Sender:
    """
    Sends data to the monitoring server.

    Parameters
    ----------
    api_key : str
        The API key to use for authentication.
    host : str, Default: 'https://api.deepchecks.com/v1'
        The hostname of the monitoring server.
    timeout : float
        The timeout for the connection to the monitoring server.
    """

    def __init__(self,
                 api_key: str,
                 host="https://api.deepchecks.com/v1",
                 timeout=200):

        self._api_key = api_key
        self._host = host
        self._timeout = timeout

        self._session = requests.Session()

        self._header = {
            "authorization": f"Bearer {self._api_key}",
        }

    def log_data(self,
                 model_id: str,
                 unique_id: Union[str, int, float],
                 model_version: str = None,
                 prediction_decision: Union[str, bool, int, float] = None,
                 label: Union[str, bool, int, float] = None,
                 features: Optional[
                    Dict[Union[str, int, float], Union[str, bool, float, int]]
                 ] = None,
                 prediction_timestamp: int = None) -> requests.Response:
        """
        Logs a record to Deepchecks monitoring API using a POST request.

        Parameters
        ----------
        model_id : str
            The unique identifier of the model.
        unique_id : Union[str, int, float]
            The unique identifier of the record.
        model_version : str, Default: None
            The version of the model.
        prediction_decision : Union[str, bool, int, float], Default: None
            The prediction decision.
        label : Union[str, bool, int, float], Default: None
            The label.
        features : Optional[
            Dict[Union[str, int, float], Union[str, bool, float, int]]
        ]
            The features.
        prediction_timestamp : int, Default: None
            The timestamp of the prediction.
        """

        # TODO: Add validations
        data = {
            "model_id": model_id,
            "unique_id": unique_id,
            "model_version": model_version,
            "prediction_decision": prediction_decision,
            "label": label,
            "features": features,
            "prediction_timestamp": prediction_timestamp,
        }

        # TODO: Make this async
        return self._session.post(
            f"{self._host}/logs",
            json=data,
            headers=self._header,
            timeout=self._timeout,
        )


