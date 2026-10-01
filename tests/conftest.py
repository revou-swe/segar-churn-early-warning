import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segar_churn.data_checks import validate_and_clean  # noqa: E402
from segar_churn.generate_data import generate  # noqa: E402


@pytest.fixture(scope="session")
def raw_small():
    """A small (150-outlet) synthetic dataset with the same defects as the full one."""
    return generate(n_outlets=150, seed=7)


@pytest.fixture(scope="session")
def clean_small(raw_small):
    clean, _ = validate_and_clean(raw_small)
    clean["outlets"]["onboard_date"] = __import__("pandas").to_datetime(clean["outlets"]["onboard_date"])
    return clean
