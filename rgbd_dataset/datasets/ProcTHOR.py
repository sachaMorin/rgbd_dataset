import os
from typing import Union
from pathlib import Path
import glob
import numpy as np
from typing import List
from natsort import natsorted
import json

from ..BaseRGBDDataset import BaseRGBDDataset
from ..utils import invert_se3


class ProcTHOR(BaseRGBDDataset):
    def __init__(self, split: str = "train", **kwargs):
        self.split = split
        self.scene = kwargs["scene"]
        self.base_path = Path(kwargs["base_path"])

        if isinstance(self.scene, int):
            self.scenes = [self.scene]
        elif self.scene == "all":
            self.scenes = natsorted(os.listdir(self.base_path / self.split ))
        else:
            raise ValueError("Invalide scene argument.")

        super().__init__(**kwargs)

    def get_rgb_paths(self) -> List[str]:
        rgb_paths = []

        for s in self.scenes:
            path_str = str(
                self.base_path / self.split / str(s) / "color_main" / "*.jpg"
            )
            rgb_paths += natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        depth_paths = []

        for s in self.scenes:
            path_str = str(
                self.base_path / self.split / str(s) / "depth_main" / "*.png"
            )
            depth_paths += natsorted(glob.glob(path_str))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        pose_paths = []

        for s in self.scenes:
            robot2cam_path = str(
                self.base_path / self.split / str(s) / "extrinsics.json"
            )
            robot2cam = np.array(
                json.loads(open(robot2cam_path).read())["color_main"]
            ).reshape(4, 4)
            cam2robot = invert_se3(robot2cam)

            pose_path = str(
                self.base_path / self.split / str(s) / "pose" / "*.json"
            )
            pose_paths += natsorted(glob.glob(pose_path))

        poses = []
        for path in pose_paths:
            robot2world = json.loads(open(path).read())
            robot2world = np.array(robot2world).reshape(4, 4)
            cam2world = robot2world @ cam2robot
            poses.append(cam2world)
        return poses
