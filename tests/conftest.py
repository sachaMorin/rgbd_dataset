import os
from pathlib import Path

import pytest

# Point this at a SemiStaticSim ``privileged`` directory (the one that contains
# ``run_0/``) to run the tests that need real data.
DATA_ENV = "SEMISTATICSIM_DATA"


@pytest.fixture(scope="session")
def privileged_dir() -> Path:
    path = os.environ.get(DATA_ENV)
    if not path:
        pytest.skip(f"set {DATA_ENV} to a SemiStaticSim privileged directory")
    path = Path(path)
    if not (path / "run_0").is_dir():
        pytest.fail(f"{DATA_ENV}={path} has no run_0/ directory")
    return path


@pytest.fixture(scope="session")
def dataset_kwargs(privileged_dir):
    return dict(
        dataset_name="semistaticsim",
        base_path=str(privileged_dir),
        scene="run_0",
        width=640,
        height=640,
        depth_scale=1000.0,
        fx=320.0,
        fy=320.0,
        cx=320.0,
        cy=320.0,
        depth_trunc=5.0,
    )
