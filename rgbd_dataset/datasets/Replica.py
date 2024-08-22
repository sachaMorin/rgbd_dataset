from typing import Union
from pathlib import Path
import glob
import numpy as np
from typing import List
from natsort import natsorted
import cv2

from ..BaseRGBDDataset import BaseRGBDDataset


class Replica(BaseRGBDDataset):
    def get_rgb_paths(self) -> List[str]:
        path_str = str(self.base_path / self.scene / "results/frame*.jpg")
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(self.base_path / self.scene / "results/depth*.png")
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(self.base_path / self.scene / "traj.txt")
        poses = []
        with open(pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_total_images):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            poses.append(c2w)
        return poses
    
    def get_intrinsic_matrices(self) -> List[np.array]:
        # Constant intrinsics
        intrinsics = np.array([
            [600., 0.0, 599.5],
            [0.0, 600., 339.5],
            [0.0, 0.0, 1.0]
        ])
        return [intrinsics] * self.num_total_images
