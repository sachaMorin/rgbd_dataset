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

DAYS_IN_WEEK = 7
HOURS_IN_DAY = 24
MINS_IN_HOUR = 60
SECS_IN_MIN = 60
HOURS_IN_WEEK = DAYS_IN_WEEK * HOURS_IN_DAY
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class Perpetua(BaseRGBDDataset):
    def __init__(
        self,
        rgb_dir: str = "rgb",
        depth_dir: str = "depth",
        pose_dir: str = "camera_pose",
        intrinsics_dir: str = "intrinsics",
        mapping_dir: str = "mapping.json",
        cloud_dir: str = "cloud.json",
        transform_dir: str = "transform.yaml",
        schedule_dir: str = "schedule.json",
        **kwargs,
    ):
        self.rgb_dir = rgb_dir
        self.pose_dir = pose_dir
        self.depth_dir = depth_dir
        self.intrinsics_dir = intrinsics_dir
        self.mapping_dir = mapping_dir
        self.cloud_dir = cloud_dir
        self.transform_dir = transform_dir
        self.schedule_dir = schedule_dir

        super().__init__(**kwargs)

        self.virtual_start_time, self.start_timestamp = self.get_virtual_start_time()
        self.timestamps = self.get_timestamps()
        self._mapping = self._load_json(self.scene, self.mapping_dir)
        self._cloud = self._load_json(self.scene, self.cloud_dir)
        self._schedule = self._load_json(self.scene, self.schedule_dir)
        self._T_global = self._load_transform_yaml(self.transform_dir) if self.transform_dir else None

    def _load_json(self, dir_name: str, path: str) -> dict:
        path = self.base_path / dir_name / path
        with open(path, "r") as f:
            return json.load(f)

    def get_virtual_start_time(self) -> str:
        path = self.base_path / self.scene / "date.txt"

        # read the date string from the file like week1_tuesday_1400
        # format: Week 1, Tuesday: 14:00
        first_line = path.read_text().splitlines()[0].strip()
        start_timestamp = float(path.read_text().splitlines()[2].strip())

        match = re.match(
            r"Week\s*(\d+),\s*([A-Za-z]+):\s*(\d{1,2}):(\d{2})", first_line
        )
        if not match:
            log.error(f"Unexpected date format: {first_line}")

        week, day, hh, mm = match.groups()
        # Map to hours and compute virtual start time
        week_int = int(week) - 1
        day_int = DAYS.index(day.lower())
        hour_int = int(hh)
        minute_int = int(mm)
        start_hour = (
            (week_int * HOURS_IN_WEEK)
            + (day_int * HOURS_IN_DAY)
            + hour_int
            + (minute_int / MINS_IN_HOUR)
        )
        return start_hour, start_timestamp

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

        rel_timestamps = [ts - self.start_timestamp for ts in timestamps]

        final_timestamps = []
        for t in rel_timestamps:
            final_timestamps.append(
                self.virtual_start_time + t / (SECS_IN_MIN * MINS_IN_HOUR)
            )

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

    def _load_transform_yaml(self, yaml_path: str):
        path = self.base_path / self.scene / yaml_path
        if not path.exists():
            return None

        with open(path, "r") as f:
            d = yaml.safe_load(f)

        qx, qy, qz, qw = d["rotation"]
        tx, ty, tz = d["translation"]

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat((qx, qy, qz, qw), scalar_first=False).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        return T

    def get_pickupable_names(self) -> List[str]:
        return list(self._mapping["P_names"])

    def get_receptacles_names(self) -> List[str]:
        return list(self._mapping["R_names"])

    def get_pickupable_to_receptacles(self) -> dict[str, List[str]]:
        return {
            k: list(v)
            for k, v in self._mapping["P_to_R_names"].items()
        }

    def _permute_axes(self, corners: np.ndarray, mode: str) -> np.ndarray:
        if mode == "xyz":
            return corners
        if mode == "yxz":
            return corners[:, [1, 0, 2]]
        if mode == "xzy":
            return corners[:, [0, 2, 1]]
        if mode == "zyx":
            return corners[:, [2, 1, 0]]
        if mode == "zxy":
            return corners[:, [2, 0, 1]]
        if mode == "yzx":
            return corners[:, [1, 2, 0]]
        raise ValueError(f"Unknown axis mode: {mode}")

    def _flip_axis(self, corners: np.ndarray, flip: str) -> np.ndarray:
        if flip == "none":
            return corners
        out = corners.copy()
        if flip == "x":
            out[:, 0] *= -1
        elif flip == "y":
            out[:, 1] *= -1
        elif flip == "z":
            out[:, 2] *= -1
        else:
            raise ValueError(f"Unknown flip: {flip}")
        return out

    def get_receptacles_bbox(self) -> dict[str, dict]:
        T = self._T_global
        AXIS_MODE = "xyz"
        FLIP = "none"

        out = {}
        for obj in self._cloud["objects"]:
            corners = np.asarray(obj["vertices"], dtype=np.float64)
            if corners.shape != (8, 3):
                raise ValueError(f"{obj.get('name')} bbox shape {corners.shape}, expected (8,3)")

            corners = self._permute_axes(corners, AXIS_MODE)
            corners = self._flip_axis(corners, FLIP)

            if T is not None:
                corners_h = np.hstack([corners, np.ones((8, 1), dtype=np.float64)])
                corners = (T @ corners_h.T).T[:, :3]

            out[obj["name"]] = {"cornerPoints": corners}

        return out
    
    def get_pickupables_bbox(self) -> dict:
        return None
    
    def get_obj_to_rec_assignment(self) -> dict:
        out = {
            obj: info["rec_name"]
            for obj, info in self._schedule["objects"].items()
        }
        return out

    def get_assignment(self) -> dict:
        objs = self._schedule["objects"]
        out = {
            p: bool(objs.get(p, {}).get("status", False))
            for p in self.get_pickupable_names()
        }
        return out

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
