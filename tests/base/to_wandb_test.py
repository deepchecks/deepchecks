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
"""to wandb tests"""
import wandb
from hamcrest import assert_that, equal_to, not_none

from deepchecks.tabular.suites import full_suite

wandb.setup(wandb.Settings(mode='disabled', program=__name__, program_relpath=__name__, disable_code=True))

def test_check_full_suite_not_failing(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    suite_res.to_wandb()
    assert_that(wandb.run, equal_to(None))

def test_check_full_suite_init_before(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    wandb.init()
    suite_res.to_wandb()
    assert_that(wandb.run, not_none())

def test_check_full_suite_deticated_false(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    suite_res.to_wandb()
    assert_that(wandb.run, not_none())
    wandb.finish()
    assert_that(wandb.run, equal_to(None))

def test_check_full_suite_kwargs(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    suite_res = full_suite().run(train, test, model)
    suite_res.to_wandb(project='ahh', config={'ahh': 'oh'})
    assert_that(wandb.run, equal_to(None))

def test_check_plotly(iris_split_dataset_and_model, simple_custom_plt_check):
    train, test, _ = iris_split_dataset_and_model

    simple_custom_plt_check.run(train, test).to_wandb()
    assert_that(wandb.run, equal_to(None))
