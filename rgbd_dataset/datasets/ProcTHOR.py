from typing import Union
from pathlib import Path
import glob
import numpy as np
from typing import List
from natsort import natsorted
import json
import cv2

from ..BaseRGBDDataset import BaseRGBDDataset


class ProcTHOR(BaseRGBDDataset):
    def __init__(self, scene_id: int, split: str = "train", **kwargs):
        self.scene_id = scene_id
        self.split = split
        super().__init__(**kwargs)

    def get_rgb_paths(self) -> List[str]:
        path_str = str(self.base_path / self.split / str(self.scene_id) / "color_main" / "*.jpg")
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(self.base_path / self.split / str(self.scene_id) / "depth_main" / "*.png")
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        robot2cam_path = str(self.base_path / self.split / str(self.scene_id) / "extrinsics.json")
        robot2cam = np.array(json.loads(open(robot2cam_path).read())["color_main"]).reshape(4, 4)
        cam2robot = np.linalg.inv(robot2cam)

        pose_path = str(self.base_path / self.split / str(self.scene_id) / "pose" / "*.json")
        pose_paths = natsorted(glob.glob(pose_path))
        poses = []
        for path in pose_paths:
            robot2world = json.loads(open(path).read())
            robot2world = np.array(robot2world).reshape(4, 4)
            cam2world = robot2world @ cam2robot
            poses.append(cam2world)
        return poses

    def read_rgb(self, path: Union[str, Path]) -> np.ndarray:
        return cv2.imread(str(path))

    def read_depth(self, path: Union[str, Path]) -> np.ndarray:
        return cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
