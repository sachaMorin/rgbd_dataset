from pathlib import Path
import hydra
from omegaconf import DictConfig

import open3d as o3d
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

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
    translations = []
    rotations = []

    progress_bar = tqdm(total=len(dataset))
    progress_bar.set_description(f"Running ICP")

    for obs in dataset:
        frame_pc = obs["point_cloud"]
        frame_pc = denoiser(frame_pc)

        if cfg.downsampling_voxel_size > 0:
            frame_pc = frame_pc.voxel_down_sample(
                voxel_size=cfg.downsampling_voxel_size
            )

        if len(pcd_scene.points):
            trans_init = prev_transform

            if cfg.color:
                # Colored ICP
                radius = cfg.downsampling_voxel_size if cfg.downsampling_voxel_size > 0 else 0.01
                pcd_scene.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius= radius * 2, max_nn=30))
                frame_pc.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

                reg_p2p = o3d.pipelines.registration.registration_colored_icp(
                    frame_pc, pcd_scene, cfg.icp.eps, trans_init,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                    relative_rmse=1e-6,
                                                                    max_iteration=cfg.icp.max_iter))

            else:
                # Standard ICP
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
            
            
            
            if cfg.debug:
                print(reg_p2p)
                print(reg_p2p.transformation)

            frame_pc = frame_pc.transform(reg_p2p.transformation)
            prev_transform = np.array(reg_p2p.transformation)
            translations += [prev_transform[:3, 3]]
            rotations += [R.from_matrix(prev_transform[:3, :3]).as_euler("xyz", degrees=True)]
            obs["camera_pose"] = prev_transform @ obs["camera_pose"] 



        if dataset.relative_pose and cfg.undo_relative_pose:
            obs["camera_pose"] = dataset.first_pose @ obs["camera_pose"]

        corrected_poses.append(obs["camera_pose"])
        pcd_scene += frame_pc

        if cfg.downsampling_voxel_size > 0:
            pcd_scene = pcd_scene.voxel_down_sample(
                voxel_size=cfg.downsampling_voxel_size
            )

        progress_bar.update(1)

    geometries += [pcd_scene]
    translations = np.stack(translations)
    rotations = np.stack(rotations)
    print(f"Average Translation: {np.mean(translations, axis=0)}")
    print(f"Std     Translation: {np.std(translations, axis=0)}")
    print(f"Average Rotation   : {np.mean(rotations, axis=0)}")
    print(f"Std     Rotation   : {np.std(rotations, axis=0)}")

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
