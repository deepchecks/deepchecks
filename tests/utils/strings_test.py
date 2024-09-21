from unittest.mock import patch

from deepchecks.utils.strings import create_new_file_name, json_encoder

import numpy as np
import pytest


@pytest.fixture
def mock_os_path_exists():
    with patch("os.path.exists") as mock_exists:
        yield mock_exists


def test_create_new_file_name_no_conflict(mock_os_path_exists):
    mock_os_path_exists.return_value = False
    new_file_name = create_new_file_name("example.txt")
    assert new_file_name == "example.txt"


def test_create_new_file_name_with_conflict(mock_os_path_exists):
    mock_os_path_exists.side_effect = [True, False]
    new_file_name = create_new_file_name("example.txt")
    assert new_file_name == "example (1).txt"


def test_create_new_file_name_with_multiple_conflicts(mock_os_path_exists):
    mock_os_path_exists.side_effect = [True, True, False]
    new_file_name = create_new_file_name("example.txt")
    assert new_file_name == "example (2).txt"


def test_create_new_file_name_no_extension_no_conflict(mock_os_path_exists):
    mock_os_path_exists.return_value = False
    new_file_name = create_new_file_name("example")
    assert new_file_name == "example.html"


def test_create_new_file_name_no_extension_with_conflict(mock_os_path_exists):
    mock_os_path_exists.side_effect = [True, False]
    new_file_name = create_new_file_name("example")
    assert new_file_name == "example (1).html"


def test_create_new_file_name_custom_suffix_no_conflict(mock_os_path_exists):
    mock_os_path_exists.return_value = False
    new_file_name = create_new_file_name("example", "md")
    assert new_file_name == "example.md"


def test_create_new_file_name_custom_suffix_with_conflict(mock_os_path_exists):
    mock_os_path_exists.side_effect = [True, False]
    new_file_name = create_new_file_name("example", "md")
    assert new_file_name == "example (1).md"


def test_json_encoder_with_numpy_int():
    obj = np.int32(42)
    assert json_encoder(obj) == 42


def test_json_encoder_with_numpy_float():
    obj = np.float64(3.14)
    assert json_encoder(obj) == 3.14


def test_json_encoder_with_numpy_array():
    obj = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        json_encoder(obj)


def test_json_encoder_with_non_numpy_object():
    obj = "string"
    assert json_encoder(obj) is None
