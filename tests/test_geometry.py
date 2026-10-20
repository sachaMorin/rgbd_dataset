import importlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from rgbd_dataset.dataloader import collate_fn_np_single, collate_fn_np_single_cb
from rgbd_dataset.rgbd_to_pcd import rgbd_to_pcd
from rgbd_dataset.utils import invert_se3


def random_se3(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = np.eye(4)
    T[:3, :3] = R.random(random_state=seed).as_matrix()
    T[:3, 3] = rng.normal(size=3)
    return T


def test_invert_se3_round_trip():
    T = random_se3(0)
    np.testing.assert_allclose(invert_se3(T) @ T, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(invert_se3(T), np.linalg.inv(T), atol=1e-12)


def test_rgbd_to_pcd_unprojects_to_world():
    h, w, fx, cx = 4, 4, 2.0, 1.5
    K = np.array([[fx, 0, cx], [0, fx, cx], [0, 0, 1.0]])
    depth = np.zeros((h, w), dtype=np.uint16)
    depth[1, 3] = 2000  # 2 m at depth_scale=1000
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    pose = random_se3(1)

    pcd = rgbd_to_pcd(rgb, depth, pose, K, w, h, depth_trunc=5.0, depth_scale=1000.0)

    z = 2.0
    p_cam = np.array([(3 - cx) * z / fx, (1 - cx) * z / fx, z, 1.0])
    np.testing.assert_allclose(np.asarray(pcd.points), [(pose @ p_cam)[:3]], atol=1e-5)


def test_collate_fn_returns_single_observation():
    obs = {"rgb": np.zeros((2, 2, 3))}
    assert collate_fn_np_single([obs]) is obs
    assert collate_fn_np_single_cb() is collate_fn_np_single
    with pytest.raises(ValueError):
        collate_fn_np_single([obs, obs])


def test_legacy_submodule_import_path_aliases_same_modules():
    legacy = importlib.import_module("rgbd_dataset.rgbd_dataset.utils")
    assert legacy is importlib.import_module("rgbd_dataset.utils")
    assert legacy.invert_se3 is invert_se3
