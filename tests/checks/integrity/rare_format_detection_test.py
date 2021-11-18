"""Contains unit tests for the rare_format_detection check."""
import re
from datetime import datetime

import numpy as np
import pandas as pd
from hamcrest import assert_that, equal_to, not_none, none, has_length, empty

from deepchecks import Dataset, ConditionCategory
from deepchecks.checks import RareFormatDetection

from tests.checks.utils import equal_condition_result
from tests.checks.utils import ANY_FLOAT_REGEXP



def util_generate_dataframe():
    np.random.seed(42)
    datelist = pd.date_range(start=datetime.today(), periods=200, freq='D', normalize=True)
    s_date = pd.Series([d.strftime('%Y-%m-%d') for d in datelist], name='date')

    emaillist = [''.join(np.random.choice(a=list('abcdefghijklmnopqrstuvwxyz'), p=[1 / 26] * 26,
                                          size=np.random.choice(a=[6, 7, 8], p=[0.2, 0.5, 0.3]))) + '@gmail.com' for x
                 in range(200)]
    s_email = pd.Series(emaillist, name='email')
    df = pd.DataFrame([s_date, s_email]).T

    return df

def fail_email(df: pd.DataFrame) -> pd.DataFrame:
    df['email'].loc[[0, 1]] = ['myname@gmail.com1', 'myname@gmail.co']
    return df



def fail_date(df: pd.DataFrame) -> pd.DataFrame:
    df['date'].loc[0:2] = [datetime.strptime(d, '%Y-%m-%d').strftime('%Y-%b-%d') for d in df['date'].loc[0:2]]
    return df


def test_assert_nothing_found():
    df = util_generate_dataframe()
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value, empty())


def test_assert_change_in_format():
    df = util_generate_dataframe()
    df['date'].loc[0:2] = [datetime.strptime(d, '%Y-%m-%d').strftime('%Y-%b-%d') for d in df['date'].loc[0:2]]
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value['date'].loc['ratio of rare samples'].values[0], equal_to('1.50% (3)'))
    assert_that(res.value.get('email'), none())


def test_assert_change_in_format2():
    df = util_generate_dataframe()
    fail_email(df)
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value.get('date'), none())
    assert_that(res.value['email'].loc['ratio of rare samples'].values[0], equal_to('1.00% (2)'))


def test_assert_param_rarity_threshold():
    df = util_generate_dataframe()
    fail_email(df)
    c = RareFormatDetection(rarity_threshold=0.01)
    res = c.run(dataset=Dataset(df))
    assert_that(res.value.get('date'), none())
    assert_that(res.value['email'].loc['ratio of rare samples'].values[0], equal_to('0.50% (1)'))


def test_assert_param_pattern_match_method():
    df = util_generate_dataframe()
    fail_email(df)
    c = RareFormatDetection(pattern_match_method='all')
    res = c.run(dataset=Dataset(df))
    assert_that(res.value.get('date'), none())
    assert_that(len(res.value['email'].columns), equal_to(5))


def test_assert_param_columns():
    df = util_generate_dataframe()
    df['stam'] = 'stam'
    fail_email(df)
    fail_date(df)
    c = RareFormatDetection(columns=['date', 'email'])
    res = c.run(dataset=Dataset(df))
    assert_that(res.value.get('email'), not_none())
    assert_that(res.value.get('date'), not_none())
    assert_that(res.value.get('stam'), none())


def test_assert_param_ignore_columns():
    df = util_generate_dataframe()
    df['stam'] = 'stam'
    fail_email(df)
    fail_date(df)
    c = RareFormatDetection(ignore_columns=['stam'])
    res = c.run(dataset=Dataset(df))
    assert_that(res.value.get('email'), not_none())
    assert_that(res.value.get('date'), not_none())
    assert_that(res.value.get('stam'),none())


def test_runs_on_numbers():
    df = pd.DataFrame(np.ones((100, 1)) * 11111, columns=['numbers'])
    df.iloc[0, 0] = 1111
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value['numbers'].loc['ratio of rare samples'].values[0], equal_to('1.00% (1)'))


def test_runs_on_mixed():
    df = pd.DataFrame(np.ones((100, 1)) * 11111, columns=['mixed'])
    df.iloc[0, 0] = 'aaaaa'
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value['mixed'].loc['ratio of rare samples'].values[0], equal_to('1.00% (1)'))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    # Arrange
    check = RareFormatDetection(n_top_columns=1)
    # Act
    result_ds = check.run(train, clf).value
    # Assert
    assert_that(result_ds, has_length(1))


def test_nan():
    df = pd.DataFrame(np.ones((101, 1)) * 11111, columns=['mixed'])
    df.iloc[0, 0] = np.nan
    df.iloc[1, 0] = 'aaaaaaa'
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value['mixed'].loc['ratio of rare samples'].values[0], equal_to('1.00% (1)'))


def test_mostly_nan():
    df = pd.DataFrame([np.nan] * 100, columns=['mixed'])
    df.iloc[0, 0] = 'aaaaa'
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value, empty())


def test_ratio_of_rare_formats_condition_that_should_pass():
    df = pd.DataFrame(np.ones((100, 1)) * 11111, columns=['mixed'])
    df.iloc[0, 0] = 'aaaaa'

    check = RareFormatDetection().add_condition_ratio_of_rare_formats_not_greater_than(0.02)
    check_result = check.run(dataset=Dataset(df))
    condition_result, *_ = check.conditions_decision(check_result)

    assert_that(
        condition_result,
        matcher=equal_condition_result( # type: ignore
            is_pass=True,
            name="Rare formats ratio is not greater than 0.02",
            details="",
            category=ConditionCategory.FAIL
        )
    )


def test_ratio_of_rare_formats_condition_that_should_not_pass():
    df = pd.DataFrame(np.ones((100, 1)) * 11111, columns=['mixed'])
    df.iloc[0, 0] = 'aaaaa'

    check = RareFormatDetection().add_condition_ratio_of_rare_formats_not_greater_than(0.002)
    check_result = check.run(dataset=Dataset(df))
    condition_result, *_ = check.conditions_decision(check_result)

    assert_that(
        condition_result,
        matcher=equal_condition_result( # type: ignore
            is_pass=False,
            name="Rare formats ratio is not greater than 0.002",
            details='Ratio of the rare formates is greater than 0.002: mixed.',
            category=ConditionCategory.FAIL
        )
    )
