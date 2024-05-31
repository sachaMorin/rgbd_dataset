import hydra
from omegaconf import DictConfig

import open3d as o3d
import numpy as np


@hydra.main(version_base=None, config_path="conf", config_name="pcd_scene")
def main(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    pcd_scene = o3d.geometry.PointCloud()
    geometries = []
    poses = []

    for obs in dataset:
        pcd_scene += obs["point_cloud"]

        if cfg.downsampling_voxel_size > 0:
            pcd_scene = pcd_scene.voxel_down_sample(voxel_size=0.05)

        if cfg.draw_cam:
            cam = o3d.geometry.LineSet.create_camera_visualization(
                obs["depth"].shape[1],
                obs["depth"].shape[0],
                obs["intrinsics"],
                np.linalg.inv(obs["camera_pose"]),  # World to cam
                scale=0.05,
            )
            geometries += [cam]

        if cfg.draw_traj:
            poses += [obs["camera_pose"][0:3, 3]]

            # Add line between this camera and previous one
            if len(poses) > 1:
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector([poses[-2], poses[-1]])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
                geometries += [line]

    geometries += [pcd_scene]

    if cfg.draw_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        geometries += [frame]

    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()
