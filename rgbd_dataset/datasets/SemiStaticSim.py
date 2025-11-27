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
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def read_parquets(parquet_path: str) -> dict:
    file_names = natsorted(glob.glob(parquet_path))
    # Read all parquet files and concatenate them
    data_frames = [pl.read_parquet(file) for file in file_names]
    concatenated_df = pl.concat(data_frames, how="vertical")
    return concatenated_df


def split_camel_preserve_acronyms(name):
    # Insert space between lowercase → uppercase
    # OR between acronym → normal word
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return s.lower()


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

    def get_pickupable_names(self) -> List[str]:
        pickupable_names_path = str(
            self.base_path / self.scene / "pickupable_names.json"
        )
        pickupable_names = json.loads(open(pickupable_names_path).read())
        # pickupable_names = [split_camel_preserve_acronyms(name.split('|')[0]) for name in pickupable_names]
        return pickupable_names

    def get_receptacles_names(self) -> List[str]:
        receptacles_names_path = str(
            self.base_path / self.scene / "receptacle_names.json"
        )
        receptacles_names = json.loads(open(receptacles_names_path).read())
        # receptacles_names = [split_camel_preserve_acronyms(name.split('|')[0]) for name in receptacles_names]
        return receptacles_names

    def get_receptacles_bbox(self) -> dict:
        receptacles_bbox_path = str(
            self.base_path / self.scene / "receptacles_aabb.json"
        )
        receptacles_bbox = json.loads(open(receptacles_bbox_path).read())

        new_receptacles_bbox = {}
        for key, value in receptacles_bbox.items():
            # object_name = split_camel_preserve_acronyms(key.split('|')[0])
            object_name = key

            for point in value["cornerPoints"]:
                point[1] = -point[1]
            value["center"]["y"] = -value["center"]["y"]

            new_receptacles_bbox[object_name] = value

        return new_receptacles_bbox

    def get_timestamps(self) -> List[float]:
        path_str = str(self.base_path / self.scene / "*.parquet")
        df = read_parquets(path_str)
        timestamps = df["_timestamp"].to_numpy()
        timestamps = timestamps[
            self.sequence_start : self.sequence_end : self.sequence_stride
        ]

        return timestamps

    def get_rgb_paths(self) -> List[str]:
        path_str = str(
            self.base_path / self.scene / self.img_dir / self.rgb_dir / "*.png"
        )
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(
            self.base_path / self.scene / self.img_dir / self.depth_dir / "*.npz"
        )
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths

    def read_depth(self, path: str) -> np.ndarray:
        depth_data = np.load(path)
        depth = depth_data["frame"].squeeze()
        depth = depth.astype(np.float32)
        return depth

    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(self.base_path / self.scene / self.pose_dir / "*.json")
        pose_paths = natsorted(glob.glob(pose_path))
        poses = []
        for path in pose_paths:
            pose = json.loads(open(path).read())

            position = pose["position"]
            rotation = pose["rotation"]

            # Intrinsic: (Z-Y'-X'') is Rot(Z)Rot(Y)Rot(X)
            # Extrinsic: (x-y-z) is Rot(Z)Rot(Y)Rot(X)
            yaw, pitch = rotation["y"], rotation["x"]
            robot2world = R.from_euler("zyx", [0.0, yaw, 0.0], degrees=True).as_matrix()
            robot2cam = R.from_euler("zyx", [0.0, 0.0, pitch], degrees=True).as_matrix()

            # The transpose
            rot_mx = robot2world @ robot2cam.T
            # This is equivalent to all operations above, the negation of the pitch
            # transforms the ai2thor frame roright-handed: x(right), y(down), z(forward)
            # rot_mx = R.from_euler('ZYX', [0.0, yaw, -pitch], degrees=True).as_matrix()
            # Hence we need to also invert the y-position for things to be consistent
            # pose_mx[0:3, 3] = [position['x'], -position['y'], position['z']]
            pose_mx = np.eye(4)
            pose_mx[0:3, 0:3] = rot_mx
            pose_mx[0:3, 3] = [position["x"], -position["y"], position["z"]]
            poses.append(pose_mx)

        return poses

    def get_semantics_paths(self) -> List[str]:
        path_str = str(
            self.base_path / self.scene / self.img_dir / self.semantics_dir / "*.png"
        )
        semantics_paths = natsorted(glob.glob(path_str))
        return semantics_paths

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
        result["timestamp"] = timestamp

        if self.rgb_transform is not None:
            result["rgb"] = self.rgb_transform(result["rgb"])

        if self.depth_transform is not None:
            result["depth"] = self.depth_transform(result["depth"])

        return result
