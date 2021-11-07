"""Contains unit tests for the rare_format_detection check."""
from datetime import datetime

import numpy as np
import pandas as pd

from mlchecks import Dataset
from mlchecks.checks import RareFormatDetection

from hamcrest import assert_that, equal_to, empty, not_none, none


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


def test_assert_nothing_found():
    df = util_generate_dataframe()
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value['date'], empty())
    assert_that(res.value['email'], empty())


def test_assert_change_in_format():
    df = util_generate_dataframe()
    df['date'].loc[0:2] = [datetime.strptime(d, '%Y-%m-%d').strftime('%Y-%b-%d') for d in df['date'].loc[0:2]]
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value['date'].loc['ratio of rare samples'].values[0], equal_to('1.50% (3)'))
    assert_that(res.value['email'], empty())


def test_assert_change_in_format2():
    df = util_generate_dataframe()
    df['email'].loc[[0, 1]] = ['myname@gmail.com1', 'myname@gmail.co']
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value['date'], empty())
    assert_that(res.value['email'].loc['ratio of rare samples'].values[0], equal_to('1.00% (2)'))


def test_assert_param_rarity_threshold():
    df = util_generate_dataframe()
    df['email'].loc[[0, 1]] = ['myname@gmail.com1', 'myname@gmail.co']
    c = RareFormatDetection(rarity_threshold=0.01)
    res = c.run(dataset=Dataset(df))
    assert_that(res.value['date'], empty())
    assert_that(res.value['email'].loc['ratio of rare samples'].values[0], equal_to('0.50% (1)'))


def test_assert_param_pattern_match_method():
    df = util_generate_dataframe()
    df['email'].loc[[0, 1]] = ['myname@gmail.com1', 'myname@gmail.co']
    c = RareFormatDetection(pattern_match_method='all')
    res = c.run(dataset=Dataset(df))
    assert_that(res.value['date'], empty())
    assert_that(len(res.value['email'].columns), equal_to(5))


def test_assert_param_columns():
    df = util_generate_dataframe()
    df['stam'] = 'stam'
    c = RareFormatDetection(columns=['date', 'email'])
    res = c.run(dataset=Dataset(df))
    assert_that(res.value.get('email'), not_none())
    assert_that(res.value.get('date'), not_none())
    assert_that(res.value.get('stam'),none())


def test_assert_param_ignore_columns():
    df = util_generate_dataframe()
    df['stam'] = 'stam'
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
