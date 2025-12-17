import hydra
from omegaconf import DictConfig

import open3d as o3d
import numpy as np

from rgbd_dataset.utils import invert_se3
from rgbd_dataset.ViserServer import ViserServer


@hydra.main(version_base=None, config_path="conf", config_name="pcd_scene")
def main(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    viser_server = ViserServer()
    pcd_scene = o3d.geometry.PointCloud()

    for obs in dataset:
        pcd_scene += obs["point_cloud"]

        if cfg.voxel_size > 0:
            pcd_scene = pcd_scene.voxel_down_sample(voxel_size=cfg.voxel_size)

        viser_server.camera_poses.append(obs["camera_pose"])
        viser_server.camera_imgs.append(obs["rgb"])

    # Append intrinsics
    viser_server.intrinsics = obs["intrinsics"]

    # Get bounding boxes
    bboxes = dataset.get_receptacles_bbox()
    for bbox in bboxes.values():
        corners = np.array(bbox["cornerPoints"], dtype=np.float64)
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(corners)
            )
        viser_server.oobb.append(bbox)

    viser_server.collate(pcd_scene)
    viser_server.display_object_rgb()
    # o3d.visualization.draw_geometries(geometries)
    while True:
        pass


if __name__ == "__main__":
    main()
