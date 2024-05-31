from typing import Union, List
from pathlib import Path
import os
import glob
import numpy as np
from typing import List
from natsort import natsorted
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R

from ..BaseRGBDDataset import BaseRGBDDataset


class Unreal(BaseRGBDDataset):
    def get_rgb_paths(self) -> List[str]:
        path_str = str(self.base_path / self.scene / "data/ColorImage/*.png")
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(self.base_path / self.scene / "data/DepthImage/*.exr")
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(self.base_path / self.scene / "CameraPoses.csv")
        df = pd.read_csv(pose_path, header=0, index_col=0)
        poses = []
        for _, row in df.iterrows():
            c2w = np.eye(4)
            rot = R.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]])
            c2w[:3, :3] = rot.as_matrix()
            c2w[:3, 3] = np.array([row["tx"], row["ty"], row["tz"]]) / 100
            poses.append(c2w)

        return poses

    def read_rgb(self, path: Union[str, Path]) -> np.ndarray:
        return cv2.imread(str(path))

    def read_depth(self, path: Union[str, Path]) -> np.ndarray:
        return cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
