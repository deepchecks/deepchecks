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

import gower
import numpy as np
import pandas as pd
from hamcrest import (assert_that, greater_than, has_item, has_length,
                      less_than_or_equal_to, close_to)

from deepchecks.utils import correlation_methods
from scipy.stats import entropy


df = pd.DataFrame({'fName': ['Noam', 'Nir', 'Nadav', 'Sol', 'Noam'],
                   'Age': [300, 24, 24, 24, 300],
                   'lName': ['Shir', 'Gabby', 'Matan', 'Shir', 'Shir'],
                   'sName': ['JKL', 'JKL', 'JKL', 'JKL', 'JKL'],
                   })


def test_conditional_entropy():
    s_fname_lname = correlation_methods.conditional_entropy(df['fName'], df['lName'])
    assert_that(s_fname_lname, close_to(0.381, 0.01))
    s_lname_fname = correlation_methods.conditional_entropy(df['lName'], df['fName'])
    assert_that(s_lname_fname, close_to(0, 0.00001))  # if we know the first names we can deduce last names
    s_fname_sname = correlation_methods.conditional_entropy(df['fName'], df['sName'])
    assert_that(s_fname_sname, close_to(1.3322, 0.01)) # sName gives no information

def test_theil_u():
    s_fname_lname = correlation_methods.theil_u_correlation(df['fName'], df['lName'])
    assert_that(s_fname_lname, close_to(0.7133204, 0.01))
    s_lname_fname = correlation_methods.theil_u_correlation(df['lName'], df['fName'])
    assert_that(s_lname_fname, close_to(1, 0.00001))  # if we know the first names we can deduce last names
    s_fname_sname = correlation_methods.theil_u_correlation(df['fName'], df['sName'])
    assert_that(s_fname_sname, close_to(0, 0.00001))  # sName gives no information
    s_fname_fname = correlation_methods.theil_u_correlation(df['fName'], df['fName'])
    assert_that(s_fname_fname, close_to(1, 0.00001))  # full correlation

