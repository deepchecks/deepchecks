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
"""The dataset module containing the recsys UserDataset class and its functions."""
# pylint: disable=inconsistent-quotes,protected-access
import typing as t

import numpy as np
import pandas as pd

from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.utils.task_type import TaskType

__all__ = ['UserDataset', 'ItemDataset', 'InteractionDataset']

TDataset = t.TypeVar('TDataset', 'InteractionDataset', 'UserDataset', 'ItemDataset')


class UserDataset(Dataset):
    """
    A custom dataset class for handling user-related data and labels.

    Parameters
    ----------
    df : Any
        The main data source, which could be a pandas DataFrame, numpy array, etc.
    label : Union[Hashable, pd.Series, pd.DataFrame, np.ndarray], optional
        Labels associated with the data, by default None.
    user_index_name : Hashable, optional
        Name of the user index, by default None.
    features : Sequence[Hashable], optional
        List of feature names, by default None.
    cat_features : Sequence[Hashable], optional
        List of categorical feature names, by default None.
    index_name : Hashable, optional
        Name of the index, by default None.
    set_index_from_dataframe_index : bool, optional
        Set the index from the DataFrame index, by default False.
    datetime_name : Hashable, optional
        Name of the datetime column, by default None.
    set_datetime_from_dataframe_index : bool, optional
        Set datetime from DataFrame index, by default False.
    convert_datetime : bool, optional
        Convert datetime columns, by default True.
    datetime_args : Dict, optional
        Additional arguments for datetime conversion, by default None.
    max_categorical_ratio : float, optional
        Maximum categorical ratio, by default 0.01.
    max_categories : int, optional
        Maximum number of categories, by default None.
    label_type : str, optional
        Type of label (e.g., "RECOMMENDATION"), by default None.
    dataset_name : str, optional
        Name of the dataset, by default None.
    label_classes : None, optional
        Label classes, by default None.
    **kwargs
        Other keyword arguments.

    Attributes
    ----------
    user_index_name : Hashable
        Get the user index name.

    Methods
    -------
    copy(new_data: pd.DataFrame) -> UserDataset
        Create a copy of the dataset with new data.
    """

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
            label_classes=None,
            **kwargs
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
            label_classes=label_classes,
            **kwargs,

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
    """
    A custom dataset class for handling item-related data.

    Parameters
    ----------
    df : Any
        The main data source, which could be a pandas DataFrame, numpy array, etc.
    item_index_name : Hashable, optional
        Name of the item index, by default None.
    item_column_name : Hashable, optional
        Name of the item column, by default None.
    features : Sequence[Hashable], optional
        List of feature names, by default None.
    cat_features : Sequence[Hashable], optional
        List of categorical feature names, by default None.
    index_name : Hashable, optional
        Name of the index, by default None.
    set_index_from_dataframe_index : bool, optional
        Set the index from the DataFrame index, by default False.
    datetime_name : Hashable, optional
        Name of the datetime column, by default None.
    set_datetime_from_dataframe_index : bool, optional
        Set datetime from DataFrame index, by default False.
    convert_datetime : bool, optional
        Convert datetime columns, by default True.
    datetime_args : Dict, optional
        Additional arguments for datetime conversion, by default None.
    max_categorical_ratio : float, optional
        Maximum categorical ratio, by default 0.01.
    max_categories : int, optional
        Maximum number of categories, by default None.
    dataset_name : str, optional
        Name of the dataset, by default None.
    **kwargs
        Other keyword arguments.

    Attributes
    ----------
    item_index_name : Hashable
        Get the item index name.
    item_index_to_ordinal : Dict[Hashable, int]
        Get a dictionary mapping item index names to ordinal values.

    Methods
    -------
    copy(new_data: pd.DataFrame) -> ItemDataset
        Create a copy of the dataset with new data.
    """

    _item_index_name: t.Optional[t.Hashable]

    def __init__(
            self,
            df: t.Any,
            item_index_name: t.Optional[t.Hashable] = None,
            item_column_name: t.Optional[t.Hashable] = None,
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
            dataset_name: t.Optional[str] = None,
            **kwargs
    ):

        super().__init__(
            df=df,
            features=features,
            cat_features=cat_features,
            label_type=TaskType.RECOMMENDETION,
            index_name=index_name,
            set_index_from_dataframe_index=set_index_from_dataframe_index,
            max_categorical_ratio=max_categorical_ratio,
            max_categories=max_categories,
            dataset_name=dataset_name,
            **kwargs,

        )
        self._item_index_name = item_index_name
        self.item_column_name = item_column_name

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


class InteractionDataset(Dataset):
    """
    A custom dataset class for handling interaction data between users and items.

    Parameters
    ----------
    df : Any
        The main data source, which could be a pandas DataFrame, numpy array, etc.
    user_index_name : Hashable, optional
        Name of the user index, by default None.
    item_index_name : Hashable, optional
        Name of the item index, by default None.
    interaction_column_name : Hashable, optional
        Name of the interaction column, by default None.
    features : Sequence[Hashable], optional
        List of feature names, by default None.
    cat_features : Sequence[Hashable], optional
        List of categorical feature names, by default None.
    index_name : Hashable, optional
        Name of the index, by default None.
    set_index_from_dataframe_index : bool, optional
        Set the index from the DataFrame index, by default False.
    datetime_name : Hashable, optional
        Name of the datetime column, by default None.
    set_datetime_from_dataframe_index : bool, optional
        Set datetime from DataFrame index, by default False.
    convert_datetime : bool, optional
        Convert datetime columns, by default True.
    datetime_args : Dict, optional
        Additional arguments for datetime conversion, by default None.
    max_categorical_ratio : float, optional
        Maximum categorical ratio, by default 0.01.
    max_categories : int, optional
        Maximum number of categories, by default None.
    dataset_name : str, optional
        Name of the dataset, by default None.
    **kwargs
        Other keyword arguments.

    Attributes
    ----------
    user_index_name : Hashable
        Get the user index name.
    item_index_name : Hashable
        Get the item index name.
    interaction_column_name : Hashable
        Get the interaction column name.
    user_index_to_ordinal : Dict[Hashable, int]
        Get a dictionary mapping user index names to ordinal values.

    Methods
    -------
    copy(new_data: pd.DataFrame) -> InteractionDataset
        Create a copy of the dataset with new data.
    __add__(other: InteractionDataset) -> InteractionDataset
        Concatenate two datasets to create a combined dataset.
    """

    _user_index_name: t.Optional[t.Hashable]
    _item_index_name: t.Optional[t.Hashable]
    _interaction_column_name: t.Optional[t.Hashable]

    def __init__(
            self,
            df: t.Any,
            user_index_name: t.Optional[t.Hashable] = None,
            item_index_name: t.Optional[t.Hashable] = None,
            interaction_column_name: t.Optional[t.Hashable] = None,
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
            **kwargs
    ):
        super().__init__(
            df=df,
            features=features,
            label_type=TaskType.RECOMMENDETION,
            cat_features=cat_features,
            index_name=index_name,
            set_index_from_dataframe_index=set_index_from_dataframe_index,
            datetime_name=datetime_name,
            set_datetime_from_dataframe_index=set_datetime_from_dataframe_index,
            convert_datetime=convert_datetime,
            datetime_args=datetime_args,
            dataset_name=dataset_name,
            **kwargs,
        )
        self._user_index_name = user_index_name
        self._item_index_name = item_index_name
        self._interaction_column_name = interaction_column_name

    def copy(self: TDataset, new_data: pd.DataFrame) -> TDataset:
        """Create a copy of the dataset with new data.

        Args:
            new_data (pd.DataFrame): New data to replace the existing data.

        Returns:
            TDataset: A new dataset object with the provided data.
        """
        dataset = super().copy(new_data)
        dataset._user_index_name = self._user_index_name
        dataset._item_index_name = self._item_index_name
        dataset._interaction_column_name = self._interaction_column_name

        return dataset

    @property
    def user_index_name(self) -> t.Optional[t.Hashable]:
        """Get the name of the user index column.

        Returns:
            t.Optional[t.Hashable]: The name of the user index column.
        """
        return self._user_index_name

    @property
    def item_index_name(self) -> t.Optional[t.Hashable]:
        """Get the name of the item index column.

        Returns:
            t.Optional[t.Hashable]: The name of the item index column.
        """
        return self._item_index_name

    @property
    def interaction_column_name(self) -> t.Optional[t.Hashable]:
        """Get the name of the interaction column.

        Returns:
            t.Optional[t.Hashable]: The name of the interaction column.
        """
        return self._interaction_column_name

    @property
    def user_index_to_ordinal(self) -> t.Optional[dict]:
        """Map user index values to ordinals.

        Returns:
            t.Optional[dict]: A mapping of user index values to their corresponding ordinals.
        """
        if self.user_index_name is None:
            return None
        return dict(zip(self.user_index_name, range(len(self.user_index_name))))

    def __add__(self, other: 'InteractionDataset') -> 'InteractionDataset':
        """Combine two datasets into a new dataset.

        Args:
            other (InteractionDataset): Another dataset to combine with.

        Returns:
            InteractionDataset: A new dataset containing the combined data.
        """
        combined_df = pd.concat([self.data, other.data], ignore_index=True)
        combined_dataset = InteractionDataset(combined_df,
                                              user_index_name=self._user_index_name,
                                              item_index_name=self._item_index_name,
                                              interaction_column_name=self._interaction_column_name,
                                              features=self.features,
                                              cat_features=self.cat_features,
                                              index_name=self.index_name,
                                              datetime_name=self.datetime_name)

        return combined_dataset
