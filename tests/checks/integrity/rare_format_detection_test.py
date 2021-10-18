"""Contains unit tests for the rare_format_detection check."""
from datetime import datetime

import numpy as np
import pandas as pd

from mlchecks import Dataset
from mlchecks.checks import RareFormatDetection

from hamcrest import assert_that, equal_to, empty


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
    assert_that(res.value['date'].loc['ratio of rare patterns to common patterns'].values[0], equal_to('1.52%'))
    assert_that(res.value['email'], empty())


def test_assert_change_in_format2():
    df = util_generate_dataframe()
    df['email'].loc[[0, 1]] = ['myname@gmail.com1', 'myname@gmail.co']
    c = RareFormatDetection()
    res = c.run(dataset=Dataset(df))
    assert_that(res.value['date'], empty())
    assert_that(res.value['email'].loc['ratio of rare patterns to common patterns'].values[0], equal_to('0.50%'))
