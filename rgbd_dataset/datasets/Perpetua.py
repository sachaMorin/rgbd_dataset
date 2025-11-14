import glob
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


class Perpetua(BaseRGBDDataset):
    def __init__(
        self,
        rgb_dir: str = "rgb",
        depth_dir: str = "depth",
        pose_dir: str = "camera_pose",
        intrinsics_dir: str = "intrinsics",
        **kwargs,
    ):
        self.rgb_dir = rgb_dir
        self.pose_dir = pose_dir
        self.depth_dir = depth_dir
        self.intrinsics_dir = intrinsics_dir

        super().__init__(**kwargs)

        self.virtual_start_time = self.get_virtual_start_time()
        self.timestamps = self.get_timestamps()

    def get_virtual_start_time(self) -> str:
        path = self.base_path / self.scene / "date.txt"

        # read the date string from the file like week1_tuesday_1400
        # format: Week 1, Tuesday: 14:00
        first_line = path.read_text().splitlines()[0].strip()

        match = re.match(
            r"Week\s*(\d+),\s*([A-Za-z]+):\s*(\d{1,2}):(\d{2})", first_line
        )
        if not match:
            log.error(f"Unexpected date format: {first_line}")

        week, day, hh, mm = match.groups()
        day = day.lower()

        virtual_time = f"week{week}_{day}_{hh}{mm}"

        return virtual_time

    def get_timestamps(self) -> List[float]:
        path_str = str(self.base_path / self.scene / self.rgb_dir / "*.jpg")
        rgb_paths = natsorted(glob.glob(path_str))
        timestamps = []
        for path in rgb_paths:
            filename = path.split("/")[-1]
            timestamp_str = filename.split(".")[0] + "." + filename.split(".")[1]
            timestamps.append(float(timestamp_str))

        timestamps = timestamps[
            self.sequence_start : self.sequence_end : self.sequence_stride
        ]

        first_timestamp = timestamps[0]
        rel_timestamps = [ts - first_timestamp for ts in timestamps]

        final_timestamps = []
        for t in rel_timestamps:
            final_timestamps.append(f"{self.virtual_start_time}_{t:.2f}")

        return final_timestamps

    def get_rgb_paths(self) -> List[str]:
        path_str = str(self.base_path / self.scene / self.rgb_dir / "*.jpg")
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(self.base_path / self.scene / self.depth_dir / "*.png")
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(self.base_path / str(self.scene) / self.pose_dir / "*.json")
        pose_paths = natsorted(glob.glob(pose_path))
        poses = []
        for path in pose_paths:
            pose = json.loads(open(path).read())
            pose_mx = np.array(pose).reshape((4, 4))
            poses.append(pose_mx)
        return poses

    def get_intrinsic_matrices(self) -> List[np.array]:
        intrinsic_path = str(
            self.base_path / str(self.scene) / self.intrinsics_dir / "*.yaml"
        )
        paths = natsorted(glob.glob(intrinsic_path))
        intrinsics = []
        for path in paths:
            with open(path, "r") as f:
                calib = yaml.safe_load(f)
            cm = calib["camera_matrix"]["data"]
            cm_mx = np.array(cm, dtype=float).reshape((3, 3))
            intrinsics.append(cm_mx)
        return intrinsics

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
            timestamp=timestamp,
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
