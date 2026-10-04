import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bvivhedge.simulate import MarketParams, simulate_market  # noqa: E402


@pytest.fixture(scope="session")
def market():
    return simulate_market(MarketParams(days=120), seed=7)


@pytest.fixture(scope="session")
def bars(market):
    return market.bars
