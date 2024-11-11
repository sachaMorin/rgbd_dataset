import glob
import numpy as np
from typing import List
from natsort import natsorted
import json
from scipy.spatial.transform import Rotation as R

from ..BaseRGBDDataset import BaseRGBDDataset


class Isaacsim2(BaseRGBDDataset):
    def __init__(
        self,
        rgb_dir: str = "rgb",
        depth_dir: str = "depth",
        pose_dir: str = "camera_pose",
        intrinsics_dir: str = "intrinsics",
        **kwargs,
    ):
        self.rgb_dir = rgb_dir
        self.pose_dir = pose_dir
        self.depth_dir = depth_dir
        self.intrinsics_dir = intrinsics_dir
        super().__init__(**kwargs)

    def get_rgb_paths(self) -> List[str]:
        path_str = str(self.base_path / self.scene / self.rgb_dir / "*.jpg")
        print(path_str)
        rgb_paths = natsorted(glob.glob(path_str))

        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(self.base_path / self.scene / self.depth_dir / "*.png")
        depth_paths = natsorted(glob.glob(path_str))

        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(self.base_path / str(self.scene) / self.pose_dir / "*.json")
        pose_paths = natsorted(glob.glob(pose_path))
        poses = []

        for path in pose_paths:
            with open(path, "r") as f:
                pose_data = json.load(f)

            translation = np.array(
                [
                    pose_data["translation"]["x"],
                    pose_data["translation"]["y"],
                    pose_data["translation"]["z"],
                ]
            )

            quaternion = pose_data["rotation"]  # [qx, qy, qz, qw] format
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            ext_rot = R.from_euler("zy", [-90, 90], degrees=True).as_matrix()
            rotation_matrix = rotation_matrix @ ext_rot

            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = translation
            poses.append(pose_matrix)

        return poses
