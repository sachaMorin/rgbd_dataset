from typing import Union
from pathlib import Path
import glob
import numpy as np
from typing import List
from natsort import natsorted
import cv2

from ..BaseRGBDDataset import BaseRGBDDataset


class Replica(BaseRGBDDataset):
    def __init__(self, scene_name: str, **kwargs):
        self.scene_name = scene_name
        super().__init__(**kwargs)

    def get_rgb_paths(self) -> List[str]:
        path_str = str(self.base_path / self.scene_name / "results/frame*.jpg")
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(self.base_path / self.scene_name / "results/depth*.png")
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(self.base_path / self.scene_name / "traj.txt")
        poses = []
        with open(pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_total_images):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            poses.append(c2w)
        return poses

    def read_rgb(self, path: Union[str, Path]) -> np.ndarray:
        return cv2.imread(str(path))

    def read_depth(self, path: Union[str, Path]) -> np.ndarray:
        return cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
