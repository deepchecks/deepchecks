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
from hamcrest import assert_that, equal_to

from deepchecks.utils.function import run_available_kwargs


def test_not_error():
    def aaa(a, b, c, f=10, m=5):
        if m is None:
            return 99
        return a + b + c + f
    assert_that(run_available_kwargs(aaa, a=2, b=3, c=5, d=10), equal_to(20))
    assert_that(run_available_kwargs(aaa, a=1, b=1, c=5, f=0), equal_to(7))
    assert_that(run_available_kwargs(aaa, a=4, b=0, c=3, f=1), equal_to(8))
    assert_that(run_available_kwargs(aaa, a=4, b=0, c=3, f=1, m=None), equal_to(99))
