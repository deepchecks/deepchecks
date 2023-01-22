# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
import re
import typing as t

from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_items, has_length, instance_of, raises
from plotly.basedatatypes import BaseFigure

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.vision.checks import ClassPerformance
from deepchecks.vision.metrics_utils import CustomClassificationScorer
from deepchecks.vision.metrics_utils.confusion_matrix_counts_metrics import AVAILABLE_EVALUATING_FUNCTIONS
from tests.base.utils import equal_condition_result


def test_coco_segmentation(segmentation_coco_visiondata_train, segmentation_coco_visiondata_test):
    # Arrange
    check = ClassPerformance(n_to_show=2, show_only='largest')

    # Act
    result = check.run(segmentation_coco_visiondata_train, segmentation_coco_visiondata_test)

    # Assert
    assert_that(set(result.value['Class']), has_length(6))
    assert_that(result.value, has_length(11))
    assert_that(result.value.iloc[0]['Value'], close_to(0.97, 0.01))


def test_mnist_largest_without_display(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = ClassPerformance(n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test, with_display=False)

    # Assert
    assert_that(set(result.value['Class']), has_length(10))
    assert_that(result.value, has_length(40))
    assert_that(result.display, has_length(0))


def test_mnist_largest(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = ClassPerformance(n_to_show=2, show_only='largest')

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    # Assert
    assert_that(result.display, has_length(1))
    figure = t.cast(BaseFigure, result.display[0])
    assert_that(figure, instance_of(BaseFigure))
    assert_that(figure.data[0]['y'][0], close_to(1.0, 0.01))
    assert_that(figure.data[0]['y'][1], close_to(0.961, 0.01))
    assert_that(figure.data[0]['hovertemplate'], equal_to('Dataset=Train<br>Metric=Precision<br>Class Name=%{x}<br>'
                                                          'sum of Value=%{y}<extra></extra>'))
    assert_that(figure.data[1]['y'][0], close_to(1.0, 0.01))
    assert_that(figure.data[1]['y'][1], close_to(1, 0.01))
    assert_that(figure.data[1]['hovertemplate'], equal_to('Dataset=Train<br>Metric=Recall<br>Class Name=%{x}<br>'
                                                          'sum of Value=%{y}<extra></extra>'))

    value = result.value
    assert_that(set(value['Class']), has_length(10))
    assert_that(value, has_length(40))
    assert_that(min(value[value['Dataset'] == 'Test']['Value']), close_to(0.9375, 0.01))
    assert_that(min(value[value['Dataset'] == 'Train']['Value']), close_to(0.869, 0.01))
    assert_that(max(value[value['Dataset'] == 'Test']['Value']), equal_to(1.0))
    assert_that(max(value[value['Dataset'] == 'Train']['Value']), equal_to(1.0))


def test_mnist_smallest(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = ClassPerformance(n_to_show=2, show_only='smallest')
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    # Assert
    assert_that(result.display, has_length(1))
    figure = t.cast(BaseFigure, result.display[0])
    assert_that(figure, instance_of(BaseFigure))
    assert_that(figure.data[0]['y'][0], close_to(1.0, 0.01))
    assert_that(figure.data[0]['y'][1], close_to(0.937, 0.01))
    assert_that(figure.data[0]['hovertemplate'], equal_to('Dataset=Train<br>Metric=Precision<br>Class Name=%{x}<br>'
                                                          'sum of Value=%{y}<extra></extra>'))
    assert_that(figure.data[1]['y'][0], close_to(1.0, 0.01))
    assert_that(figure.data[1]['y'][1], close_to(1, 0.01))
    assert_that(figure.data[1]['hovertemplate'], equal_to('Dataset=Train<br>Metric=Recall<br>Class Name=%{x}<br>'
                                                          'sum of Value=%{y}<extra></extra>'))

    value = result.value
    assert_that(set(value['Class']), has_length(10))
    assert_that(value, has_length(40))
    assert_that(min(value[value['Dataset'] == 'Test']['Value']), close_to(0.9375, 0.01))
    assert_that(min(value[value['Dataset'] == 'Train']['Value']), close_to(0.869, 0.01))
    assert_that(max(value[value['Dataset'] == 'Test']['Value']), equal_to(1.0))
    assert_that(max(value[value['Dataset'] == 'Train']['Value']), equal_to(1.0))


def test_mnist_alt(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    scorers = {'p': CustomClassificationScorer('precision_per_class'),
               'r': CustomClassificationScorer('recall_per_class')}
    check = ClassPerformance(n_to_show=2, show_only='smallest', scorers=scorers) \
        .add_condition_test_performance_greater_than(1)
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    # Assert
    assert_that(result.display, has_length(1))
    figure = t.cast(BaseFigure, result.display[0])
    assert_that(figure, instance_of(BaseFigure))
    assert_that(figure.data[0]['y'][0], close_to(1.0, 0.01))
    assert_that(figure.data[0]['y'][1], close_to(0.937, 0.01))
    assert_that(figure.data[0]['hovertemplate'], equal_to('Dataset=Train<br>Metric=p<br>Class Name=%{x}<br>'
                                                          'sum of Value=%{y}<extra></extra>'))
    assert_that(figure.data[1]['y'][0], close_to(1.0, 0.01))
    assert_that(figure.data[1]['y'][1], close_to(1, 0.01))
    assert_that(figure.data[1]['hovertemplate'], equal_to('Dataset=Train<br>Metric=r<br>Class Name=%{x}<br>'
                                                          'sum of Value=%{y}<extra></extra>'))

    value = result.value
    assert_that(set(value['Class']), has_length(10))
    assert_that(value, has_length(40))
    assert_that(min(value[value['Dataset'] == 'Test']['Value']), close_to(0.9375, 0.01))
    assert_that(min(value[value['Dataset'] == 'Train']['Value']), close_to(0.869, 0.01))
    assert_that(max(value[value['Dataset'] == 'Test']['Value']), equal_to(1.0))
    assert_that(max(value[value['Dataset'] == 'Train']['Value']), equal_to(1.0))

    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=False,
            details=re.compile(
                r'Found 20 scores below threshold.\nFound minimum score for r metric of value 0.93 for class \d.'),
            name='Scores are greater than 1'
        )
    ))


def test_coco_best(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    check = ClassPerformance(n_to_show=2, show_only='best')
    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test)

    # Assert
    assert_that(result.display, has_length(1))
    figure = t.cast(BaseFigure, result.display[0])
    assert_that(figure, instance_of(BaseFigure))
    assert_that(figure.data[0]['y'][0], close_to(0.23465, 0.01))
    assert_that(figure.data[0]['hovertemplate'], equal_to('Dataset=Train<br>Metric=Average Precision<br>Class Name=%{x}'
                                                          '<br>sum of Value=%{y}<extra></extra>'))
    assert_that(figure.data[1]['y'][0], close_to(0.233, 0.01))
    assert_that(figure.data[1]['hovertemplate'], equal_to('Dataset=Train<br>Metric=Average Recall<br>Class Name=%{x}'
                                                          '<br>sum of Value=%{y}<extra></extra>'))

    value = result.value
    assert_that(set(value['Class']), has_length(71))
    assert_that(value, has_length(244))
    assert_that(max(value[value['Dataset'] == 'Test']['Value']), equal_to(1))
    assert_that(max(value[value['Dataset'] == 'Train']['Value']), equal_to(1))
    assert_that(min(value[value['Dataset'] == 'Test']['Value']), equal_to(0))
    assert_that(min(value[value['Dataset'] == 'Train']['Value']), equal_to(0))
    assert_that(value[value['Dataset'] == 'Train']['Value'].mean(), close_to(0.402, 0.001))
    assert_that(value[value['Dataset'] == 'Test']['Value'].mean(), close_to(0.405, 0.001))


def test_class_list(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = ClassPerformance(class_list_to_show=[1])
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    # Assert
    assert_that(result.display, has_length(1))
    figure = t.cast(BaseFigure, result.display[0])
    assert_that(figure, instance_of(BaseFigure))
    assert_that(figure.data[0]['y'][0], close_to(1, 0.01))
    assert_that(figure.data[0]['y'], has_length(1))
    assert_that(figure.data[0]['hovertemplate'], equal_to('Dataset=Train<br>Metric=Recall<br>Class Name=%{x}<br>'
                                                          'sum of Value=%{y}<extra></extra>'))
    assert_that(figure.data[1]['y'][0], close_to(0.962, 0.01))
    assert_that(figure.data[1]['y'], has_length(1))
    assert_that(figure.data[1]['hovertemplate'], equal_to('Dataset=Train<br>Metric=Precision<br>Class Name=%{x}<br>'
                                                          'sum of Value=%{y}<extra></extra>'))
    value = result.value
    assert_that(set(value['Class']), has_length(10))
    assert_that(value, has_length(40))
    assert_that(min(value[value['Dataset'] == 'Test']['Value']), close_to(0.9375, 0.01))
    assert_that(min(value[value['Dataset'] == 'Train']['Value']), close_to(0.869, 0.01))
    assert_that(max(value[value['Dataset'] == 'Test']['Value']), equal_to(1.0))
    assert_that(max(value[value['Dataset'] == 'Train']['Value']), equal_to(1.0))


def test_condition_test_performance_greater_than_pass(mnist_visiondata_train,
                                                      mnist_visiondata_test):
    # Arrange
    check = ClassPerformance().add_condition_test_performance_greater_than(0.5)

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=True,
                               details=re.compile(r'Found minimum score for Recall metric of value 0.93 for class \d.'),
                               name='Scores are greater than 0.5'))
                )


def test_condition_train_test_relative_degradation_less_than_pass(
        mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = ClassPerformance().add_condition_train_test_relative_degradation_less_than(0.1)

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=True,
            details=r'Found max degradation of 7.14% for metric Recall and class 1.',
            name='Train-Test scores relative degradation is less than 0.1'
        )
    ))


def test_condition_class_performance_imbalance_ratio_less_than(
        mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = ClassPerformance().add_condition_class_performance_imbalance_ratio_less_than(0.5, 'Precision')

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=True,
            name='Relative ratio difference between labels \'Precision\' score is less than 50%',
            details='Relative ratio difference between highest and lowest in Test dataset classes '
                    'is 4.76%, using Precision metric. Lowest class - 6: 0.95; Highest class - 0: '
                    '1\nRelative ratio difference between highest and lowest in Train dataset '
                    'classes is 6.25%, using Precision metric. Lowest class - 8: 0.94; Highest class'
                    ' - 0: 1'
        )
    ))


def test_condition_class_performance_imbalance_ratio_less_than_fail(
        mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = ClassPerformance() \
        .add_condition_class_performance_imbalance_ratio_less_than(0.0001, 'Precision')

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=False,
                               details='Relative ratio difference between highest and lowest in Test dataset classes '
                                       'is 4.76%, using Precision metric. Lowest class - 6: 0.95; Highest class - 0: '
                                       '1\nRelative ratio difference between highest and lowest in Train dataset '
                                       'classes is 6.25%, using Precision metric. Lowest class - 8: 0.94; Highest class'
                                       ' - 0: 1',
                               name='Relative ratio difference between labels \'Precision\' score is less than 0.01%'))
                )


def test_coco_thershold_scorer_list_strings(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    scorers = [name + '_per_class' for name in AVAILABLE_EVALUATING_FUNCTIONS.keys()]
    check = ClassPerformance(scorers=scorers)
    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test)
    # Assert
    assert_that(result.value, has_length(590))
    assert_that(result.display, has_length(greater_than(0)))
    assert_that(set(result.value['Metric']), equal_to(set(AVAILABLE_EVALUATING_FUNCTIONS.keys())))


def test_coco_deepchecks_scorer_list_strings_averaging(coco_visiondata_train, coco_visiondata_test):
    for avg_method in ['macro', 'micro', 'weighted']:
        # Arrange
        scorers = [name + '_' + avg_method for name in AVAILABLE_EVALUATING_FUNCTIONS.keys()]
        check = ClassPerformance(scorers=scorers)
        # Act
        result = check.run(coco_visiondata_train, coco_visiondata_test)
        # Assert
        assert_that(result.value, has_length(10))
        assert_that(result.display, has_length(greater_than(0)))
        assert_that(set(result.value['Metric']), equal_to(set(scorers)))


def test_mnist_sklearn_scorer(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = ClassPerformance(scorers={'f1': 'f1_per_class', 'recall': 'recall_per_class'})

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    # Assert
    assert_that(result.value, has_length(40))
    assert_that(result.display, has_length(greater_than(0)))
    assert_that(set(result.value['Metric']), equal_to({'f1', 'recall'}))


def test_coco_unsupported_scorers(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    check = ClassPerformance(scorers=['fnr_per_class', 'r3'])
    # Act
    assert_that(
        calling(check.run).with_args(coco_visiondata_train, coco_visiondata_test),
        raises(DeepchecksNotSupportedError,
               r'Unsupported metric: r3 of type str was given.')
    )


def test_mnist_unsupported_sklearn_scorers(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = ClassPerformance(scorers={'f1': 'f1_per_class', 'recall': 'recall_per_class', 'R3': 'r3'})
    # Act
    assert_that(
        calling(check.run
                ).with_args(mnist_visiondata_train, mnist_visiondata_test),
        raises(DeepchecksValueError,
               pattern=r'Scorer name r3 is unknown. See metric guide for a list of allowed scorer names.')
    )


def test_coco_bad_value_type_scorers(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    check = ClassPerformance(scorers={'r2': 2})
    # Act
    assert_that(
        calling(check.run
                ).with_args(coco_visiondata_train, coco_visiondata_test),
        raises(DeepchecksValueError,
               r'Excepted metric type one of \[ignite.Metric, callable, str\], was int.')
    )
