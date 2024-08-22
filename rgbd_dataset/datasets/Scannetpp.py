from pathlib import Path
import glob
import numpy as np
from typing import List
from natsort import natsorted
import json

from ..BaseRGBDDataset import BaseRGBDDataset


class Scannetpp(BaseRGBDDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_resized = (
            True  # Always resize because of depth has much lower resolution
        )

    def get_rgb_paths(self) -> List[str]:
        path_str = str(
            self.base_path / "data" / self.scene / "iphone" / "rgb" / "frame*.jpg"
        )
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(
            self.base_path / "data" / self.scene / "iphone" / "depth" / "frame*.png"
        )
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(
            self.base_path / "data" / self.scene / "iphone" / "pose_intrinsic_imu.json"
        )
        with open(pose_path, "r") as f:
            data = json.load(f)
        keys = natsorted([k for k in data.keys()])

        poses = [np.array(data[k]["aligned_pose"]).reshape((4, 4)) for k in keys]

        return poses

    def get_intrinsic_matrices(self) -> List[np.array]:
        pose_path = str(
            self.base_path / "data" / self.scene / "iphone" / "pose_intrinsic_imu.json"
        )
        with open(pose_path, "r") as f:
            data = json.load(f)
        keys = natsorted([k for k in data.keys()])

        poses = [np.array(data[k]["intrinsic"]) for k in keys]

        return poses
