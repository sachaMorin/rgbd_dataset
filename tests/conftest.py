import os
from pathlib import Path

import pytest

# Point this at a SemiStaticSim ``privileged`` directory (the one that contains the
# ``run_<hour>/`` folders) to run the tests that need real data.
DATA_ENV = "SEMISTATICSIM_DATA"


@pytest.fixture(scope="session")
def privileged_dir() -> Path:
    path = os.environ.get(DATA_ENV)
    if not path:
        pytest.skip(f"set {DATA_ENV} to a SemiStaticSim privileged directory")
    path = Path(path)
    if not any(path.glob("run_*")):
        pytest.fail(f"{DATA_ENV}={path} has no run_<hour>/ folders")
    return path


@pytest.fixture(scope="session")
def scene(privileged_dir) -> str:
    runs = sorted(privileged_dir.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
    for run in runs:
        if any((run / "images" / "rgb_first_view").glob("*.jpg")):
            return run.name
    pytest.fail(f"no run in {privileged_dir} has RGB frames")


@pytest.fixture(scope="session")
def dataset_kwargs(privileged_dir, scene):
    return dict(
        dataset_name="semistaticsim",
        base_path=str(privileged_dir),
        scene=scene,
        width=640,
        height=640,
        depth_scale=1000.0,
        fx=320.0,
        fy=320.0,
        cx=320.0,
        cy=320.0,
        depth_trunc=5.0,
    )
