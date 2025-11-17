import glob
import os
import polars as pl
import numpy as np
from typing import List
from natsort import natsorted
import json
import yaml
from scipy.spatial.transform import Rotation as R
import cv2
import re

from ..BaseRGBDDataset import BaseRGBDDataset
from ..rgbd_to_pcd import rgbd_to_pcd

import logging

log = logging.getLogger(__name__)

DAYS_IN_WEEK = 7
HOURS_IN_DAY = 24
MINS_IN_HOUR = 60
SECS_IN_MIN = 60
HOURS_IN_WEEK = DAYS_IN_WEEK * HOURS_IN_DAY
DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

def read_parquets(parquet_path: str) -> dict:
    file_names = natsorted(glob.glob(parquet_path))
    # Read all parquet files and concatenate them
    data_frames = [pl.read_parquet(file) for file in file_names]
    concatenated_df = pl.concat(data_frames, how="vertical")
    return concatenated_df

class SemiStaticSim(BaseRGBDDataset):
    def __init__(
        self,
        img_dir: str = "images",
        rgb_dir: str = "rgb_first_view",
        depth_dir: str = "depth",
        semantics_dir: str = "semantics",
        pose_dir: str = "poses",
        **kwargs,
    ):
        self.img_dir = img_dir
        self.rgb_dir = rgb_dir
        self.pose_dir = pose_dir
        self.depth_dir = depth_dir
        self.semantics_dir = semantics_dir

        super().__init__(**kwargs)

        self.semantics_paths = self.get_semantics_paths()

        self.timestamps = self.get_timestamps()

    def get_virtual_start_time(self, start_time: float) -> str:
        week = int(start_time // HOURS_IN_WEEK + 1)  # Data is in hours
        day = DAYS[int((start_time % HOURS_IN_WEEK) // HOURS_IN_DAY)]
        # Get start min and hour 
        hour = int(np.floor(start_time % HOURS_IN_DAY))
        minute = int((start_time * MINS_IN_HOUR) % MINS_IN_HOUR)

        virtual_time = f"week{week}_{day}_{hour:02d}{minute:02d}"
        return virtual_time


    def get_timestamps(self) -> List[float]:
        path_str = str(self.base_path / f"run_{self.scene}" / "*.parquet")
        df = read_parquets(path_str)
        timestamps = df["_timestamp"].to_numpy()
        # Get virtual start time 
        virtual_start_time = self.get_virtual_start_time(timestamps[0].item())
        timestamps = timestamps[
            self.sequence_start : self.sequence_end : self.sequence_stride
        ]

        # Map hours to seconds 
        timestamps = timestamps * SECS_IN_MIN * MINS_IN_HOUR

        final_timestamps = []
        for t in timestamps:
            final_timestamps.append(f"{virtual_start_time}_{t:.2f}")

        return final_timestamps

    def get_rgb_paths(self) -> List[str]:
        path_str = str(self.base_path / f"run_{self.scene}" / self.img_dir / self.rgb_dir / "*.png")
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(self.base_path / f"run_{self.scene}" / self.img_dir / self.depth_dir / "*.npz")
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths
    
    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(self.base_path / f"run_{self.scene}" / self.pose_dir / "*.json")
        pose_paths = natsorted(glob.glob(pose_path))
        poses = []
        for path in pose_paths:
            pose = json.loads(open(path).read())
            position = pose["position"]
            rotation = pose["rotation"]
            rot_mx = R.from_euler('xyz', [rotation['x'], rotation['y'], rotation['z']], degrees=True).as_matrix()
            pose_mx = np.eye(4)
            pose_mx[0:3, 0:3] = rot_mx
            pose_mx[0:3, 3] = [position['x'], position['y'], position['z']]
            poses.append(pose_mx)
        return poses

    def get_semantics_paths(self) -> List[str]:
        path_str = str(self.base_path / f"run_{self.scene}" / self.img_dir / self.semantics_dir / "*.png")
        semantics_paths = natsorted(glob.glob(path_str))
        return semantics_paths

    def get_intrinsic_matrices(self) -> List[np.array]:
        intrinsic_path = str(self.base_path / f"run_{self.scene}"  / "camera_intrinsics.json")
        paths = glob.glob(intrinsic_path)
        cam_intrinsics = json.loads(open(paths[0]).read())
        cam_matrix = np.eye(3)
        cam_matrix[0, 0] = cam_intrinsics['fx']
        cam_matrix[1, 1] = cam_intrinsics['fy']
        cam_matrix[0, 2] = cam_intrinsics['cx']
        cam_matrix[1, 2] = cam_intrinsics['cy']
        intrinsics = []
        # This already accounts for the stride
        intrinsics = [cam_matrix] * len(self.rgb_paths)
        return intrinsics

    def read_depth(self, path: str) -> np.ndarray:
        depth_data = np.load(path)
        depth = depth_data["frame"].squeeze()
        depth = (depth * self.depth_scale).astype(np.uint16)
        return depth # or depth.T?

    def __getitem__(self, idx):
        rgb = self.read_rgb(self.rgb_paths[idx])
        depth = self.read_depth(self.depth_paths[idx])
        pose = self.se3_poses[idx]
        intrinsics = self.rescale_intrinsics(self.intrinsics[idx])
        timestamp = self.timestamps[idx]

        if rgb.shape[0] != self.resized_height or rgb.shape[1] != self.resized_width:
            rgb = cv2.resize(
                rgb,
                (self.resized_width, self.resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
        if depth.shape[0] != self.resized_height or depth.shape[1] != self.resized_width:
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
        result["timestamp"] = timestamp

        if self.rgb_transform is not None:
            result["rgb"] = self.rgb_transform(result["rgb"])

        if self.depth_transform is not None:
            result["depth"] = self.depth_transform(result["depth"])

        return result
