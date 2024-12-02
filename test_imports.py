import unittest


class TestImports(unittest.TestCase):
    def test_imports(self):
        try:
            from deepchecks.tabular import Dataset  # noqa: F401
            from deepchecks.tabular.suites import data_integrity, model_evaluation, train_test_validation  # noqa: F401
        except ImportError as e:
            self.fail(f"Import failed: {e}")


if __name__ == "__main__":
    unittest.main()
