import glob
import numpy as np
from typing import List
from natsort import natsorted
import json
from scipy.spatial.transform import Rotation as R

from ..BaseRGBDDataset import BaseRGBDDataset


class RGBD(BaseRGBDDataset):
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
            pose = json.loads(open(path).read())
            pose = np.array(pose).reshape((4, 4))
            poses.append(pose)
        return poses

    def get_intrinsic_matrices(self) -> List[np.array]:
        intrinsic_path = str(
            self.base_path / str(self.scene) / self.intrinsics_dir / "*.json"
        )
        paths = natsorted(glob.glob(intrinsic_path))
        intrinsics = []
        for path in paths:
            mx = json.loads(open(path).read())
            intrinsics.append(np.array(mx).reshape((3, 3)))
        return intrinsics
