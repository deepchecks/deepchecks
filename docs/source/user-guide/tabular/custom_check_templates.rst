.. _custom_check_templates:

======================
Custom Check Templates
======================

We provide a template for each check type in order to help you bootstrap faster new custom checks. For more detailed
info about the inner workings of a custom check see the
:doc:`guide </user-guide/tabular/tutorials/plot_add_a_custom_check>`

* `Single Dataset Check <#single-dataset-check>`__
* `Train Test Check <#train-test-check>`__
* `Model Only Check <#model-only-check>`__


Single Dataset Check
--------------------------

.. code-block::

  import pandas as pd

  from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
  from deepchecks.tabular import SingleDatasetCheck, Dataset, Context

  class SingleDatasetCustomCheck(SingleDatasetCheck):
      """Description of the check. The name of the check will be the class name split by upper case letters."""

      # OPTIONAL: we can add different properties in the init
      def __init__(self, prop_a: str, prop_b: str, **kwargs):
          super().__init__(**kwargs)
          self.prop_a = prop_a
          self.prop_b = prop_b

      def run_logic(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
          # Get the dataset by its type (train/test)
         dataset: Dataset = context.get_data_by_kind(dataset_kind)
          # Get from the dataset the data
          data: pd.DataFrame = dataset.data

          # LOGIC HERE - possible to add validations on inputs and properties like prop_a and prop_b
          failing_rows = some_calc_fn(self.prop_a, self.prop_b)

          # Define result value: Adding any info that we might want to know later
          result = {
              'ratio': failing_rows.shape[0] / data.shape[0],
              'indices': failing_rows.index
          }

          # Define result display: list of either plotly-figure/dataframe/html
          display = [
              # Showing in the display only sample of 5 rows
              failing_rows[:5]
          ]

          return CheckResult(result, display=display)

      # OPTIONAL: add condition to check
      def add_condition_ratio_less_than(self, threshold: float = 0.01):
          # Define condition function: the function accepts as input the result value we defined in the run_logic
          def condition(result):
              ratio = result['ratio']
              category = ConditionCategory.PASS if ratio < threshold else ConditionCategory.FAIL
              message = f'Found X ratio of {ratio}'
              return ConditionResult(category, message)

          # Define the name of the condition
          name = f'Custom check ratio is less than {threshold}'
          # Now add it on the class instance
          return self.add_condition(name, condition)


Train Test Check
-----------------

.. code-block::

  import pandas as pd

  from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
  from deepchecks.tabular import TrainTestCheck, Dataset, Context


  class TrainTestCustomCheck(TrainTestCheck):
      """Description of the check. The name of the check will be the class name split by upper case letters."""

      # OPTIONAL: we can add different properties in the init
      def __init__(self, prop_a: str, prop_b: str, **kwargs):
          super().__init__(**kwargs)
          self.prop_a = prop_a
          self.prop_b = prop_b

      def run_logic(self, context: Context) -> CheckResult:
          # Get the 2 datasets
          train_dataset: Dataset = context.train
          test_dataset: Dataset = context.test
          # Get from the datasets the data
          train_df: pd.DataFrame = train_dataset.data
          test_df: pd.DataFrame = test_dataset.data

          # LOGIC HERE - possible to add validations on inputs and properties like prop_a and prop_b
          test_failing_rows = some_calc_fn(self.prop_a, self.prop_b)

          # Define result value: Adding any info that we might want to know later
          result = {
              'ratio': test_failing_rows.shape[0] / test_df.shape[0],
              'indices': test_failing_rows.index
          }

          # Define result display: list of either plotly-figure/dataframe/html
          display = [
              # Showing in the display only sample of 5 rows
              test_failing_rows[:5]
          ]

          return CheckResult(result, display=display)

      # OPTIONAL: add condition to check
      def add_condition_ratio_less_than(self, threshold: float = 0.01):
          # Define condition function: the function accepts as input the result value we defined in the run_logic
          def condition(result):
              ratio = result['ratio']
              category = ConditionCategory.PASS if ratio < threshold else ConditionCategory.FAIL
              message = f'Found X ratio of {ratio}'
              return ConditionResult(category, message)

          # Define the name of the condition
          name = f'Custom check ratio is less than {threshold}'
          # Now add it on the class instance
          return self.add_condition(name, condition)


Model Only Check
-------------------

.. code-block::

  from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
  from deepchecks.tabular import ModelOnlyCheck, Context


  class ModelOnlyCustomCheck(ModelOnlyCheck):
      """Description of the check. The name of the check will be the class name split by upper case letters."""

      # OPTIONAL: we can add different properties in the init
      def __init__(self, prop_a: str, prop_b: str, **kwargs):
          super().__init__(**kwargs)
          self.prop_a = prop_a
          self.prop_b = prop_b

      def run_logic(self, context: Context) -> CheckResult:
          # Get the model
          model = context.model

          # LOGIC HERE - possible to add validations on inputs and properties like prop_a and prop_b
          some_score = some_calc_fn(model, self.prop_a, self.prop_b)

          # Define result value: Adding any info that we might want to know later
          result = some_score

          # Define result display: list of either plotly-figure/dataframe/html, or Nothing if we have no display
          display = None

          return CheckResult(result, display=display)

      # OPTIONAL: add condition to check
      def add_condition_score_more_than(self, threshold: float = 1):
          # Define condition function: the function accepts as input the result value we defined in the run_logic
          def condition(result):
              category = ConditionCategory.PASS if result > 1 else ConditionCategory.FAIL
              message = f'Found X score of {result}'
              return ConditionResult(category, message)

          # Define the name of the condition
          name = f'Custom check score is more than {threshold}'
          # Now add it on the class instance
          return self.add_condition(name, condition)
