from mlchecks.base import CheckSuite
from mlchecks.checks.overview.model_info import ModelInfo
from src.checks.decisions import threshold

simple_suite = CheckSuite(
    ModelInfo()
)

