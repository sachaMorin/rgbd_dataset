import glob
import numpy as np
from typing import List
from natsort import natsorted

from ..BaseRGBDDataset import BaseRGBDDataset

class HM3D(BaseRGBDDataset):
    """Adapted from the HOVSG code at https://github.com/hovsg/HOV-SG/tree/6d9cd24d7d3b877896b00b7f1c97c2be86cdfa04"""

    def get_rgb_paths(self) -> List[str]:
        path_str = str(
            self.base_path / self.scene /  "rgb" / "*.png"
        )
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(
            self.base_path / self.scene /  "depth" / "*.png"
        )
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(
            self.base_path / self.scene /  "pose" / "*.txt"
        )

        pose_path = natsorted(glob.glob(pose_path))
        poses = []

        for path in pose_path:
            with open(path, "r") as file:
                line = file.readline().strip()
                values = line.split()
                values = [float(val) for val in values]
                transformation_matrix = np.array(values).reshape((4, 4))
                C = np.eye(4)
                C[1, 1] = -1
                C[2, 2] = -1
                transformation_matrix = np.matmul(transformation_matrix, C)
                poses.append(transformation_matrix)

        return poses

