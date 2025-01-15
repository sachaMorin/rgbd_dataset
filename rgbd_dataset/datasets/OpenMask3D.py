import glob
import numpy as np
from typing import List, Union
from natsort import natsorted
from pathlib import Path
from ..BaseRGBDDataset import BaseRGBDDataset
from ..utils import invert_se3
import math
import os


def get_number_of_images(poses_path):
    i = 0
    while os.path.isfile(os.path.join(poses_path, str(i) + ".txt")):
        i += 1
    return i


class OpenMask3D(BaseRGBDDataset):
    def __init__(
        self,
        rgb_dir: str = "rgb",
        depth_dir: str = "depth",
        pose_dir: str = "camera_pose",
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 319.5,
        cy: float = 239.5,
        intrinsics_resolution: List[int] = [640, 480],
        **kwargs,
    ):
        self.rgb_dir = rgb_dir
        self.pose_dir = pose_dir
        self.depth_dir = depth_dir

        self.intrinsic_original_resolution = intrinsics_resolution
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.intrinsic = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        super().__init__(**kwargs)

    def get_rgb_paths(self) -> List[str]:
        path = self.base_path / self.scene / self.rgb_dir
        rgb_paths = natsorted(
            [str(p) for p in Path(path).glob("*.jpg") if p.stem.isdigit()]
        )
        print("Number of images: ", len(rgb_paths))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path = self.base_path / self.scene / self.depth_dir
        depth_paths = natsorted(
            [str(p) for p in Path(path).glob("*.png") if p.stem.isdigit()]
        )
        print("Number of depths: ", len(depth_paths))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        path = self.base_path / self.scene / self.pose_dir
        pose_paths = natsorted(
            [str(p) for p in Path(path).glob("*.txt") if p.stem.isdigit()]
        )
        poses = []
        for path in pose_paths:
            pose = np.loadtxt(path)
            pose_mx = np.array(pose).reshape((4, 4))
            # pose_mx = invert_se3(pose_mx)
            poses.append(pose_mx)
        print("Number of poses: ", len(poses))
        return poses

    def get_intrinsic_matrices(self) -> List[np.array]:
        # Constant intrinsics
        self.intrinsic = self.get_adapted_intrinsic([self.height, self.width])
        print(f"Instrinsic: {self.intrinsic}")
        return [self.intrinsic] * self.num_total_images

    def get_adapted_intrinsic(self, desired_resolution):
        """Get adjusted camera intrinsics."""
        if self.intrinsic_original_resolution == desired_resolution:
            return self.intrinsic

        resize_width = int(
            math.floor(
                desired_resolution[1]
                * float(self.intrinsic_original_resolution[0])
                / float(self.intrinsic_original_resolution[1])
            )
        )

        adapted_intrinsic = self.intrinsic.copy()
        adapted_intrinsic[0, 0] *= float(resize_width) / float(
            self.intrinsic_original_resolution[0]
        )
        adapted_intrinsic[1, 1] *= float(desired_resolution[1]) / float(
            self.intrinsic_original_resolution[1]
        )
        adapted_intrinsic[0, 2] *= float(desired_resolution[0] - 1) / float(
            self.intrinsic_original_resolution[0] - 1
        )
        adapted_intrinsic[1, 2] *= float(desired_resolution[1] - 1) / float(
            self.intrinsic_original_resolution[1] - 1
        )
        return adapted_intrinsic
