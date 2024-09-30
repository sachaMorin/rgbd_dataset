from typing import Union, List, Callable
from pathlib import Path

import json
import cv2
import numpy as np
from torch.utils.data import Dataset

from .utils import invert_se3
from .rgbd_to_pcd import rgbd_to_pcd


class BaseRGBDDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
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
        fx: float = None,
        fy: float = None,
        cx: float = None,
        cy: float = None,
        rgb_transform: Union[Callable, None] = None,
        depth_transform: Union[Callable, None] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
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

        # Optional parameters for global intrinsics
        # Override get_intrinsics_matrices for per frame intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Torch transforms
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

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
        if self.sequence_end == -1:
            self.sequence_end = self.num_total_images
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
        self.first_pose = self.se3_poses[0]
        self.first_pose_inv = invert_se3(self.first_pose)

    def __len__(self):
        return len(self.rgb_paths)

    @property
    def name(self):
        return self.dataset_name + "_" + self.scene

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
        # Constant intrinsics
        intrinsics = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        return [intrinsics] * self.num_total_images

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

        if rgb.shape[0] != self.resized_height or rgb.shape[1] != self.resized_width:
            rgb = cv2.resize(
                rgb,
                (self.resized_width, self.resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
        if (
            depth.shape[0] != self.resized_height
            or depth.shape[1] != self.resized_width
        ):
            depth = cv2.resize(
                depth,
                (self.resized_width, self.resized_height),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.relative_pose:
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

        if self.rgb_transform is not None:
            result["rgb"] = self.rgb_transform(result["rgb"])

        if self.depth_transform is not None:
            result["depth"] = self.depth_transform(result["depth"])

        return result

    def to_nerfstudio_config(self, dir: str):
        config = dict(
            camera_model="OPENCV",
            w=self.width,  # Not self.resized_width. Nerfstudio reads the files directly
            h=self.height,
            frames=[],
        )

        for idx in range(len(self)):
            rgb_path = self.rgb_paths[idx]
            intrinsics = self.intrinsics[idx]  # No rescale
            pose = self.se3_poses[idx]

            if self.relative_pose:
                pose = np.dot(self.first_pose_inv, pose)

            # Switch to OpenGL convention
            pose[:, 1] *= -1
            pose[:, 2] *= -1

            fx, fy, cx, cy = (
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            )
            data = dict(
                fl_x=fx,
                fl_y=fy,
                cx=cx,
                cy=cy,
                file_path=rgb_path,
                transform_matrix=pose.tolist(),
            )
            config["frames"].append(data)

        with open(dir + "/transforms.json", "w") as f:
            json.dump(config, f)
