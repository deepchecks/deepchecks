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
"""The dataset module containing the recsys RecDataset class and its functions."""
# pylint: disable=inconsistent-quotes,protected-access
import typing as t

import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.utils.task_type import TaskType

__all__ = ['RecDataset', 'ItemDataset']

TDataset = t.TypeVar('TDataset', bound='RecDataset')


class RecDataset(Dataset):

    _user_index_name: t.Optional[t.Hashable]

    def __init__(
            self,
            df: t.Any,
            label: t.Union[t.Hashable, pd.Series, pd.DataFrame, np.ndarray] = None,
            user_index_name: t.Optional[t.Hashable] = None,
            features: t.Optional[t.Sequence[t.Hashable]] = None,
            cat_features: t.Optional[t.Sequence[t.Hashable]] = None,
            index_name: t.Optional[t.Hashable] = None,
            set_index_from_dataframe_index: bool = False,
            datetime_name: t.Optional[t.Hashable] = None,
            set_datetime_from_dataframe_index: bool = False,
            convert_datetime: bool = True,
            datetime_args: t.Optional[t.Dict] = None,
            max_categorical_ratio: float = 0.01,
            max_categories: int = None,
            label_type: str = None,
            dataset_name: t.Optional[str] = None,
            label_classes=None
    ):

        super().__init__(
            df=df,
            label=label,
            features=features,
            cat_features=cat_features,
            index_name=index_name,
            set_index_from_dataframe_index=set_index_from_dataframe_index,
            datetime_name=datetime_name,
            set_datetime_from_dataframe_index=set_datetime_from_dataframe_index,
            convert_datetime=convert_datetime,
            datetime_args=datetime_args,
            max_categorical_ratio=max_categorical_ratio,
            max_categories=max_categories,
            label_type=label_type or TaskType.RECOMMENDETION,
            dataset_name=dataset_name,
            label_classes=label_classes
        )
        self._user_index_name = user_index_name

    def copy(self: TDataset, new_data: pd.DataFrame) -> TDataset:
        """Create a copy of this Dataset with new data.

        Parameters
        ----------
        new_data (DataFrame): new data from which new dataset will be created

        Returns
        -------
        Dataset
            new dataset instance
        """
        dataset = super().copy(new_data)
        dataset._user_index_name = self.user_index_name
        return dataset

    @property
    def user_index_name(self) -> t.Optional[t.Hashable]:
        """Get the user index name."""
        return self._user_index_name


class ItemDataset(Dataset):

    _item_index_name: t.Optional[t.Hashable]

    def __init__(
            self,
            df: t.Any,
            item_index_name: t.Optional[t.Hashable] = None,
            features: t.Optional[t.Sequence[t.Hashable]] = None,
            cat_features: t.Optional[t.Sequence[t.Hashable]] = None,
            index_name: t.Optional[t.Hashable] = None,
            set_index_from_dataframe_index: bool = False,
            max_categorical_ratio: float = 0.01,
            max_categories: int = None,
            dataset_name: t.Optional[str] = None,
    ):

        super().__init__(
            df=df,
            features=features,
            cat_features=cat_features,
            index_name=index_name,
            set_index_from_dataframe_index=set_index_from_dataframe_index,
            max_categorical_ratio=max_categorical_ratio,
            max_categories=max_categories,
            dataset_name=dataset_name,
        )
        self._item_index_name = item_index_name

    def copy(self: TDataset, new_data: pd.DataFrame) -> TDataset:
        """Create a copy of this Dataset with new data.

        Parameters
        ----------
        new_data (DataFrame): new data from which new dataset will be created

        Returns
        -------
        Dataset
            new dataset instance
        """
        dataset = super().copy(new_data)
        dataset._item_index_name = self._item_index_name
        return dataset

    @property
    def item_index_name(self) -> t.Optional[t.Hashable]:
        """Get the user index name."""
        return self._item_index_name

    @property
    def item_index_to_ordinal(self) -> t.Optional[t.Dict[t.Hashable, int]]:
        """Get the user index name."""
        if self.item_index_name is None:
            return None
        return dict(zip(self.item_index_name, range(len(self.item_index_name))))
