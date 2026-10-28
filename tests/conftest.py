import numpy as np
import pandas as pd
import pytest

from bess_ops.config import ltsa_from_dict, site_from_dict, warranty_from_dict
from bess_ops.schema import site_frame


@pytest.fixture(scope="session")
def demo():
    """The synthetic battery, analysed once for all integration tests (about 30 s)."""
    import tempfile
    from pathlib import Path

    from bess_ops.study import analyse_demo
    with tempfile.TemporaryDirectory() as tmp:
        sim, frame, res = analyse_demo(Path(tmp))
    return sim, frame, res


@pytest.fixture
def site_cfg():
    return site_from_dict({"power_mw": 10, "energy_mwh": 20, "timezone": "Europe/Warsaw", "cod": "2025-01-01",
                           "aux_metering": "behind_meter"})


@pytest.fixture
def warranty(site_cfg):
    return warranty_from_dict({}, site_cfg)


@pytest.fixture
def ltsa(site_cfg):
    return ltsa_from_dict({}, site_cfg)


def hourly_frame(data: dict, start="2025-06-02 00:00", tz="Europe/Warsaw") -> pd.DataFrame:
    n = len(next(iter(data.values())))
    idx = pd.date_range(pd.Timestamp(start).tz_localize(tz), periods=n, freq="1h")
    return site_frame({k: np.asarray(v, float) for k, v in data.items()}, idx, tz)
