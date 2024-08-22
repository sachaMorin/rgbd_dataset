from pathlib import Path
import hydra
from omegaconf import DictConfig

import open3d as o3d
import numpy as np
import json
import os


class DBSCAN:
    def __init__(self, eps=0.02, min_points=10):
        self.eps = eps
        self.min_points = min_points

    def __call__(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if len(pcd.points) < self.min_points:
            return pcd

        labels = pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points)
        mask = np.array(labels) != -1

        return pcd.select_by_index(np.where(mask)[0])


@hydra.main(version_base=None, config_path="conf", config_name="icp_pose_correction")
def main(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    pcd_scene = o3d.geometry.PointCloud()
    denoiser = (
        DBSCAN(cfg.dbscan.eps, cfg.dbscan.min_points)
        if hasattr(cfg, "dbscan")
        else lambda x: x
    )
    geometries = []
    corrected_poses = []
    prev_transform = np.eye(4)

    for obs in dataset:
        frame_pc = obs["point_cloud"]
        frame_pc = denoiser(frame_pc)

        if len(pcd_scene.points):
            trans_init = prev_transform
            reg_p2p = o3d.pipelines.registration.registration_icp(
                frame_pc,
                pcd_scene,
                cfg.icp.eps,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=cfg.icp.max_iter
                ),
            )
            print(reg_p2p)

            frame_pc = frame_pc.transform(reg_p2p.transformation)
            prev_transform = np.array(reg_p2p.transformation)
            obs["camera_pose"] = prev_transform @ obs["camera_pose"]

        corrected_poses.append(obs["camera_pose"])
        pcd_scene += frame_pc

        if cfg.downsampling_voxel_size > 0:
            pcd_scene = pcd_scene.voxel_down_sample(
                voxel_size=cfg.downsampling_voxel_size
            )

    geometries += [pcd_scene]

    if cfg.show:
        o3d.visualization.draw_geometries(geometries)

    # Save corrected poses
    base_path = Path(dataset.base_path) / dataset.scene / "camera_pose_corrected"
    print(f"Saving corrected poses to {base_path}...")
    os.makedirs(base_path, exist_ok=True)
    file_names = [Path(f).stem + ".json" for f in dataset.rgb_paths]
    for f, p in zip(file_names, corrected_poses):
        path = base_path / f
        with open(path, "w") as file:
            json.dump(p.tolist(), file)


if __name__ == "__main__":
    main()
