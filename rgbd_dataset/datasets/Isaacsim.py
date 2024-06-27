from typing import Union
from pathlib import Path
import glob
import numpy as np
from typing import List
from natsort import natsorted
import cv2
import json
from scipy.spatial.transform import Rotation as R

from ..BaseRGBDDataset import BaseRGBDDataset


class Isaacsim(BaseRGBDDataset):
    def get_rgb_paths(self) -> List[str]:
        result = []
        for obj in self.scene:
            path_str = str(self.base_path / obj / "rgb/frame*.jpg")
            result += natsorted(glob.glob(path_str))
        return result

    def get_depth_paths(self) -> List[str]:
        result = []
        for obj in self.scene:
            path_str = str(self.base_path / obj / "depth/frame*.tif")
            result += natsorted(glob.glob(path_str))
        return result

    def get_se3_poses(self) -> List[np.array]:
        result = []
        for obj in self.scene:
            path_str = str(self.base_path / obj / "camera_pose/frame*.json")
            pose_paths = natsorted(glob.glob(path_str))

            poses = []

            for p in pose_paths:
                pose_list = json.load(open(p))
                poses.append(np.array(pose_list))


            # Change Isaacsim coordinate sim
            for p in poses:
                rot = R.from_euler("zy", [-90, 90],  degrees=True).as_matrix()
                p[:3, :3]  = p[:3, :3] @ rot

            result += poses


        return result
