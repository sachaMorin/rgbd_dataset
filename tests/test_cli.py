import subprocess
import sys


def test_viewer_config_resolves_from_installed_package(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "rgbd_dataset.pcd_scene", "--cfg", "job"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "_target_: rgbd_dataset.datasets.SemiStaticSim.SemiStaticSim" in out
    assert "voxel_size" in out
    assert not any(tmp_path.iterdir())  # the viewer writes no Hydra output dirs
