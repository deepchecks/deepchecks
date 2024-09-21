from deepchecks import Dataset

import numpy as np
import pandas as pd
import pytest


def test_dataset_initialization_with_dataframe():
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    dataset = Dataset(df=data)
    assert isinstance(dataset.data, pd.DataFrame)


def test_dataset_initialization_with_label_series():
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    label = pd.Series([0, 1, 1], name="target")
    dataset = Dataset(df=data, label=label)
    assert dataset.label_name == "target"
    assert "target" in dataset.data.columns


def test_dataset_initialization_with_label_column_name():
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "label": [0, 1, 1]})
    dataset = Dataset(df=data, label="label")
    assert dataset.label_name == "label"
    assert "label" in dataset.data.columns


def test_dataset_initialization_with_numpy_array():
    data = np.array([[1, 4], [2, 5], [3, 6]])
    dataset = Dataset.from_numpy(data, columns=["A", "B"])
    assert isinstance(dataset.data, pd.DataFrame)


def test_dataset_sample():
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    dataset = Dataset(df=data)
    sampled = dataset.sample(n_samples=2)
    assert len(sampled.data) == 2


def test_dataset_drop_na_labels():
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, 6], "label": [0, np.nan, 1]})
    dataset = Dataset(df=data, label="label")
    cleaned = dataset.drop_na_labels()
    assert not cleaned.data["label"].isna().any()


def test_dataset_train_test_split():
    data = pd.DataFrame({"A": range(100), "B": range(100, 200)})
    dataset = Dataset(df=data)
    train, test = dataset.train_test_split(train_size=0.7, random_state=42)
    assert len(train.data) == 70
    assert len(test.data) == 25


def test_dataset_assert_features():
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    dataset = Dataset(df=data, features=["A"])
    dataset.assert_features()
    assert dataset.features == ["A"]


def test_dataset_assert_index():
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    dataset = Dataset(df=data, index_name="A")
    dataset.assert_index()
    assert dataset.index_name == "A"


def test_dataset_assert_datetime():
    data = pd.DataFrame({"A": pd.date_range("20230101", periods=3), "B": [4, 5, 6]})
    dataset = Dataset(df=data, datetime_name="A")
    dataset.assert_datetime()
    assert dataset.datetime_name == "A"


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {"feature1": [1, 2, 3, 4, 5], "feature2": ["A", "B", "C", "D", "E"], "label": [10, 20, 30, 40, 50]}
    )


def test_repr(sample_dataframe):
    dataset = Dataset(df=sample_dataframe, label="label")
    repr_str = repr(dataset)
    assert isinstance(repr_str, str)
    assert "Dataset" in repr_str
    assert "label" in repr_str
    assert "feature1" in repr_str
    assert "feature2" in repr_str

    # Additional tests for more comprehensive checks
    assert "Column" in repr_str
    assert "DType" in repr_str
    assert "Kind" in repr_str
    assert "Additional Info" in repr_str
    assert "label" in repr_str
    assert "feature1" in repr_str
    assert "feature2" in repr_str
