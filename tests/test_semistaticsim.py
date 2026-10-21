from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import rgbd_dataset
from rgbd_dataset.datasets.SemiStaticSim import SemiStaticSim

CONF_DIR = Path(rgbd_dataset.__file__).parent / "conf"


@pytest.fixture(scope="module")
def dataset(dataset_kwargs):
    return SemiStaticSim(**dataset_kwargs, sequence_stride=50, point_cloud=True)


def test_observation_contents(dataset):
    assert len(dataset) > 0
    obs = dataset[0]
    assert obs["rgb"].shape == (640, 640, 3) and obs["rgb"].dtype == np.uint8
    assert obs["depth"].shape == (640, 640)
    assert np.nanmax(obs["depth"]) < 100  # converted to metres
    assert obs["semantics"].shape == (640, 640, 3)
    assert obs["camera_pose"].shape == (4, 4)
    R = obs["camera_pose"][:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert obs["intrinsics"].shape == (3, 3)
    assert len(obs["point_cloud"].points) > 0
    assert np.isfinite(float(obs["timestamp"]))


def test_timestamps_increase(dataset):
    ts = np.asarray(dataset.timestamps, dtype=np.float64)
    assert len(ts) == len(dataset)
    assert np.all(np.diff(ts) > 0)


def test_scene_metadata(dataset):
    pickupables = dataset.get_pickupable_names()
    receptacles = dataset.get_receptacles_names()
    assert len(pickupables) > 0 and len(receptacles) > 0

    assignment = dataset.get_assignment()
    assert set(assignment) == set(pickupables)
    assert all(isinstance(v, bool) for v in assignment.values())

    for bboxes in (dataset.get_receptacles_bbox(), dataset.get_pickupables_bbox()):
        assert len(bboxes) > 0
        for bbox in bboxes.values():
            assert np.asarray(bbox["cornerPoints"]).shape == (8, 3)

    p2r = dataset.get_pickupable_to_receptacles()
    assert set(p2r) <= set(pickupables)


def test_resize_rescales_intrinsics(dataset_kwargs, dataset):
    small = SemiStaticSim(
        **dataset_kwargs,
        sequence_stride=50,
        resized_width=320,
        resized_height=320,
        point_cloud=False,
    )
    obs = small[0]
    assert obs["rgb"].shape[:2] == obs["depth"].shape == obs["semantics"].shape[:2] == (320, 320)
    np.testing.assert_allclose(obs["intrinsics"][:2], dataset[0]["intrinsics"][:2] / 2)


def test_keyframes_select_listed_frames(dataset_kwargs, privileged_dir, scene):
    if not (privileged_dir / scene / "keyframes.txt").is_file():
        pytest.skip(f"{scene} has no keyframes.txt")
    full = SemiStaticSim(**dataset_kwargs, point_cloud=False)
    kf = SemiStaticSim(**dataset_kwargs, keyframes=True, point_cloud=False)
    idx = kf.keyframe_indices
    assert len(kf) == len(idx) > 0
    assert kf.rgb_paths == [full.rgb_paths[i] for i in idx]
    np.testing.assert_array_equal(kf.timestamps, np.asarray(full.timestamps)[idx])


def test_hydra_config_instantiates(privileged_dir, scene):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(
            config_name="pcd_scene",
            overrides=[f"dataset.base_path={privileged_dir}", f"dataset.scene={scene}", "dataset.sequence_stride=100"],
        )
    ds = instantiate(cfg.dataset)
    assert isinstance(ds, SemiStaticSim) and len(ds) > 0


def test_viser_server_displays_scene(dataset):
    import open3d as o3d

    from rgbd_dataset.ViserServer import ViserServer

    server = ViserServer()
    try:
        pcd = o3d.geometry.PointCloud()
        for obs in dataset:
            pcd += obs["point_cloud"]
            server.camera_poses.append(obs["camera_pose"])
            server.camera_imgs.append(obs["rgb"])
        server.intrinsics = obs["intrinsics"]
        for bbox in dataset.get_receptacles_bbox().values():
            server.oobb.append(
                o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(np.asarray(bbox["cornerPoints"], dtype=np.float64))
                )
            )
        server.collate(pcd)
        server.display_object_rgb()
        server.display_camera_poses()
        server.display_boxes()
        server.display_origin()
        server.on_point_size_change(None)
        assert len(server.point_cloud_handles) == 1
        assert len(server.frames) == len(dataset)
        assert len(server.box_handles) == len(server.oobb)
    finally:
        server.server.stop()
