from pathlib import Path
import glob
import numpy as np
from typing import List
from natsort import natsorted
import json
from scipy.spatial.transform import Rotation

from ..BaseRGBDDataset import BaseRGBDDataset
from ..utils import invert_se3


# This uses the ARK Kit Pose. See the other class for the COLMAP pose
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


class COLMAP_image_txt:
    def __init__(self, path):
        self.file = open(str(path), "r")

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.file).split()
        while not len(line) or line[0] == "#" or not line[-1].endswith(".jpg"):
            line = next(self.file).split()

        # Some scenes have a leading video/ in the path for no apparent reason
        line[-1] = line[-1].replace("video/", "")

        return line


class ScannetppCOLMAP(BaseRGBDDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_resized = (
            True  # Always resize because of depth has much lower resolution
        )

    def get_rgb_paths(self) -> List[str]:
        rgb_paths = []
        lines = COLMAP_image_txt(self.base_path / "data" / self.scene / "iphone" / "colmap" / "images.txt") 
        for line in lines:
            rgb_paths.append(self.base_path / "data" / self.scene / "iphone" / "rgb" / line[-1])

        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        depth_paths = []
        lines = COLMAP_image_txt(self.base_path / "data" / self.scene / "iphone" / "colmap" / "images.txt") 
        for line in lines:
            file_name = line[-1].replace("jpg", "png")
            depth_paths.append(self.base_path / "data" / self.scene / "iphone" / "depth" / file_name)

        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        se3_poses = []
        lines = COLMAP_image_txt(self.base_path / "data" / self.scene / "iphone" / "colmap" / "images.txt") 
        for line in lines:
            quat = np.array(line[1:5]).astype(float)
            translation = np.array(line[5:8]).astype(float)
            se3 = np.eye(4)
            se3[:3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
            se3[:3, 3] = translation
            se3_poses.append(invert_se3(se3))

        return se3_poses