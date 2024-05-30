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
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
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
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.sequence_start = sequence_start
        self.sequence_end = sequence_end
        self.sequence_stride = sequence_stride
        self.relative_pose = relative_pose
        self.depth_scale = depth_scale
        self.point_cloud = point_cloud
        self.depth_trunc = depth_trunc

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
        self.first_pose_inv = np.linalg.inv(self.se3_poses[0])

        # Intrinsics
        self.is_resized = self.resized_width > 0 and self.resized_height > 0
        self.intrinsic_mx = self.get_intrinsics(self.fx, self.fy, self.cx, self.cy)
        self.scaled_intrinsic_mx = self.get_scaled_intrinsics(
            self.fx, self.fy, self.cx, self.cy
        )

    def __len__(self):
        return len(self.rgb_paths)

    def get_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w_ratio: float = 1.0,
        h_ratio: float = 1.0,
    ) -> np.ndarray:
        return np.array(
            [
                [fx * w_ratio, 0, cx * w_ratio],
                [0, fy * h_ratio, cy * h_ratio],
                [0, 0, 1],
            ]
        )

    def get_scaled_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        if self.is_resized:
            w_ratio, h_ratio = (
                self.resized_width / self.width,
                self.resized_height / self.height,
            )
        else:
            w_ratio, h_ratio = 1.0, 1.0
        return self.get_intrinsics(fx, fy, cx, cy, w_ratio, h_ratio)

    def get_rgb_paths(self) -> List[str]:
        raise NotImplementedError

    def get_depth_paths(self) -> List[str]:
        raise NotImplementedError

    def get_se3_poses(self) -> List[np.array]:
        raise NotImplementedError

    def read_rgb(self, path: Union[str, Path]) -> np.ndarray:
        raise NotImplementedError

    def read_depth(self, path: Union[str, Path]) -> np.ndarray:
        raise NotImplementedError

    def check_image_size(self, img: np.ndarray):
        if img.shape[0] != self.height or img.shape[1] != self.width:
            raise ValueError(
                f"Expected image of size ({self.height}, {self.width}, ...), but got {img.shape}"
            )

    def __getitem__(self, idx):
        rgb = self.read_rgb(self.rgb_paths[idx])
        depth = self.read_depth(self.depth_paths[idx])
        pose = self.se3_poses[idx]

        self.check_image_size(rgb)
        self.check_image_size(depth)

        if self.is_resized:
            rgb = cv2.resize(
                rgb,
                (self.resized_width, self.resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
            depth = cv2.resize(
                depth,
                (self.resized_width, self.resized_height),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.relative_pose:
            pose = np.dot(pose, self.first_pose_inv)

        result = dict(
            rgb=rgb,
            depth=depth,
            extrinsics=pose,
            intrinsics=self.scaled_intrinsic_mx,
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
