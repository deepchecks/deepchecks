.. _vision__custom_check_templates:

======================
Custom Check Templates
======================

This page supplies templates for the different types of custom checks that you can create using the deepchecks package.
For more information on custom checks, please see the
:doc:`Custom Check Guide. </user-guide/vision/auto_tutorials/plot_custom_checks>`


Templates:

* `Single Dataset Check <#single-dataset-check>`__
* `Train Test Check <#train-test-check>`__
* `Model Only Check <#model-only-check>`__


Single Dataset Check
--------------------------
Check type for cases when running on a single dataset and optional model, for example integrity checks. When in suite
if 2 datasets are supplied it will run on both independently.

.. code-block::

  from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
  from deepchecks.vision import SingleDatasetCheck, Context, VisionData, Batch


  class SingleDatasetCustomCheck(SingleDatasetCheck):
      """Description of the check. The name of the check will be the class name split by upper case letters."""

      # OPTIONAL: we can add different properties in the init
      def __init__(self, prop_a: str, prop_b: str, **kwargs):
          super().__init__(**kwargs)
          self.prop_a = prop_a
          self.prop_b = prop_b

      def initialize_run(self, context: Context, dataset_kind: DatasetKind):
          # Initialize cache
          self.cache = {}
          # OPTIONAL: add validations on inputs and properties like prop_a and prop_b

      def update(self, context: Context, batch: BatchWrapper, dataset_kind: DatasetKind):
          # Get the VisionData by its type (train/test)
          dataset: VisionData = context.get_data_by_kind(dataset_kind)
          # Take from the batch the data I need it and save it on the cache
          batch_data_dict = some_calc_on_batch(batch, dataset)
          # Save the data on the cache
          self.cache.update(batch_data_dict)

      def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
          # LOGIC HERE
          failing_samples = some_calc_on_cache(self.cache, self.prop_a, self.prop_b)

          # Define result value: Adding any info that we might want to know later
          result = {
              'ratio': len(failing_samples) / len(self.cache),
              'indices': failing_samples.keys()
          }

          # Define result display: list of either plotly-figure/dataframe/html
          display = None

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
Check type for cases when running on two datasets and optional model, for example drift checks.


.. code-block::

  from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
  from deepchecks.vision import TrainTestCheck, Context, VisionData, Batch


  class SingleDatasetCustomCheck(TrainTestCheck):
      """Description of the check. The name of the check will be the class name split by upper case letters."""

      # OPTIONAL: we can add different properties in the init
      def __init__(self, prop_a: str, prop_b: str, **kwargs):
          super().__init__(**kwargs)
          self.prop_a = prop_a
          self.prop_b = prop_b

      def initialize_run(self, context: Context):
          # Initialize cache
          self.cache = {
              DatasetKind.TRAIN: {},
              DatasetKind.TEST: {}
          }
          # OPTIONAL: add validations on inputs and properties like prop_a and prop_b

      def update(self, context: Context, batch: BatchWrapper, dataset_kind: DatasetKind):
          # Get the VisionData by its type (train/test)
          dataset: VisionData = context.get_data_by_kind(dataset_kind)
          # Take from the batch the data I need it and save it on the cache
          batch_data_dict = some_calc_on_batch(batch, dataset)
          # Save the data on the cache
          self.cache[dataset_kind].update(batch_data_dict)

      def compute(self, context: Context) -> CheckResult:
          # Get the VisionData
          train_vision_data: VisionData = context.train
          test_vision_data: VisionData = context.test

          # LOGIC HERE
          failing_samples = some_calc_on_cache(self.cache, self.prop_a, self.prop_b)

          # Define result value: Adding any info that we might want to know later
          result = {
              'ratio': len(failing_samples) / len(self.cache),
              'indices': failing_samples.keys()
          }

          # Define result display: list of either plotly-figure/dataframe/html
          display = None

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
Check type for cases when running only on a model, for example model parameters check.


.. code-block::

  from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
  from deepchecks.vision import ModelOnlyCheck, Context


  class ModelOnlyCustomCheck(ModelOnlyCheck):
      """Description of the check. The name of the check will be the class name split by upper case letters."""

      # OPTIONAL: we can add different properties in the init
      def __init__(self, prop_a: str, prop_b: str, **kwargs):
          super().__init__(**kwargs)
          self.prop_a = prop_a
          self.prop_b = prop_b

      def compute(self, context: Context) -> CheckResult:
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
