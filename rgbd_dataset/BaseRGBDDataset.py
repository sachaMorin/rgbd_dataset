from typing import Union, List
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from .rgbd_to_pcd import rgbd_to_pcd


class BaseRGBDDataset(Dataset):
    def __init__(
        self,
        base_path: Union[str, Path],
        scene: str,
        width: int,
        height: int,
        resized_width: int = -1,
        resized_height: int = -1,
        sequence_start: int = 0,
        sequence_end: int = -1,
        sequence_stride: int = 1,
        relative_pose: bool = False,
        depth_scale: float = 1.0,
        point_cloud: bool = True,
        depth_trunc: float = 8.0,
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.scene = scene
        self.width = width
        self.height = height
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.sequence_start = sequence_start
        self.sequence_end = sequence_end
        self.sequence_stride = sequence_stride
        self.relative_pose = relative_pose
        self.depth_scale = depth_scale
        self.point_cloud = point_cloud
        self.depth_trunc = depth_trunc

        # Parameters for rescaling intrinsics
        if self.resized_width == -1:
            self.resized_width = self.width
        if self.resized_height == -1:
            self.resized_height = self.height
        self.w_ratio = self.resized_width / self.width
        self.h_ratio = self.resized_height / self.height

        # Get and slice paths and poses
        self.rgb_paths = self.get_rgb_paths()
        self.num_total_images = len(self.rgb_paths)
        self.rgb_paths = self.rgb_paths[
            self.sequence_start : self.sequence_end : self.sequence_stride
        ]
        self.depth_paths = self.get_depth_paths()[
            self.sequence_start : self.sequence_end : self.sequence_stride
        ]
        self.se3_poses = self.get_se3_poses()[
            self.sequence_start : self.sequence_end : self.sequence_stride
        ]
        self.intrinsics = self.get_intrinsic_matrices()[
            self.sequence_start : self.sequence_end : self.sequence_stride
        ]
        self.first_pose_inv = np.linalg.inv(self.se3_poses[0])

    def __len__(self):
        return len(self.rgb_paths)

    def rescale_intrinsics(self, intrinsics: np.ndarray):
        rescaled_intrinsics = intrinsics.copy()
        rescaled_intrinsics[0, 0] *= self.w_ratio
        rescaled_intrinsics[0, 2] *= self.w_ratio
        rescaled_intrinsics[1, 1] *= self.h_ratio
        rescaled_intrinsics[1, 2] *= self.h_ratio
        return rescaled_intrinsics

    def get_rgb_paths(self) -> List[str]:
        raise NotImplementedError

    def get_depth_paths(self) -> List[str]:
        raise NotImplementedError

    def get_se3_poses(self) -> List[np.array]:
        # Camera to world transform
        raise NotImplementedError

    def get_intrinsic_matrices(self) -> List[np.array]:
        # Camera intrinsics for each frame
        raise NotImplementedError

    def read_rgb(self, path: Union[str, Path]) -> np.ndarray:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def read_depth(self, path: Union[str, Path]) -> np.ndarray:
        return cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)

    def __getitem__(self, idx):
        rgb = self.read_rgb(self.rgb_paths[idx])
        depth = self.read_depth(self.depth_paths[idx])
        pose = self.se3_poses[idx]
        intrinsics = self.rescale_intrinsics(self.intrinsics[idx])

        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            rgb = cv2.resize(
                rgb,
                (self.resized_width, self.resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(
                depth,
                (self.resized_width, self.resized_height),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.relative_pose:
            # TODO: TRIPLE CHECK THIS
            # pose = np.dot(pose, self.first_pose_inv)
            pose = np.dot(self.first_pose_inv, pose)

        result = dict(
            rgb=rgb,
            depth=depth,
            camera_pose=pose,
            intrinsics=intrinsics,
        )

        if self.point_cloud:
            result["point_cloud"] = rgbd_to_pcd(
                **result,
                width=self.resized_width,
                height=self.resized_height,
                depth_trunc=self.depth_trunc,
                depth_scale=self.depth_scale,
            )

        result["depth"] = result["depth"] / self.depth_scale

        return result
