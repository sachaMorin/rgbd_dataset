import glob
import numpy as np
from typing import List
from natsort import natsorted
import json
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import cv2

import os

# Disable GPU memory pre-allocation to avoid OOM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Import JAX and other libraries after setting the environment variable
import jax
import jax.numpy as jnp

from ..BaseRGBDDataset import BaseRGBDDataset
from ..rgbd_to_pcd import rgbd_to_pcd
from ..data_utils import (
    load_sssd,
    GeneratedSemiStaticData,
)

import logging

log = logging.getLogger(__name__)

LHS_TO_RHS = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
RHS_TO_ROS = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


class SemiStaticSim(BaseRGBDDataset):
    def __init__(
        self,
        img_dir: str = "images",
        rgb_dir: str = "rgb_first_view",
        depth_dir: str = "depth",
        semantics_dir: str = "semantics",
        pose_dir: str = "poses",
        cam_offset: float = 0.675,
        keyframes: bool = False,
        **kwargs,
    ):
        self.img_dir = img_dir
        self.rgb_dir = rgb_dir
        self.pose_dir = pose_dir
        self.cam_offset = cam_offset
        self.depth_dir = depth_dir
        self.semantics_dir = semantics_dir
        self.keyframes = keyframes
        # If using keyframes, set stride to 1
        kwargs["sequence_stride"] = 1 if keyframes else kwargs["sequence_stride"]

        super().__init__(**kwargs)

        self.sssd_data: GeneratedSemiStaticData = load_sssd(self.base_path / self.scene)
        if self.keyframes:
            self.keyframe_indices = self.get_keyframes()
            all_semantics = self.get_semantics_paths()
            full_timestamps = self.get_timestamps() 
            # list comprehension to pick exactly the indices we want
            self.rgb_paths = [self.rgb_paths[i] for i in self.keyframe_indices]
            self.depth_paths = [self.depth_paths[i] for i in self.keyframe_indices]
            self.se3_poses = [self.se3_poses[i] for i in self.keyframe_indices]
            self.intrinsics = [self.intrinsics[i] for i in self.keyframe_indices]
            self.semantics_paths = [all_semantics[i] for i in self.keyframe_indices]
            self.timestamps = full_timestamps[self.keyframe_indices]
            self.num_total_images = len(self.rgb_paths)
        else:
            self.semantics_paths = self.get_semantics_paths()[
                self.sequence_start : self.sequence_end : self.sequence_stride
            ]
            self.timestamps = self.get_timestamps()
        self.assignment = self.get_assignment()

    def get_keyframes(self) -> bool:
        path_str = str(self.base_path / self.scene /  "keyframes.txt")
        with open(path_str, "r") as f:
            keyframe_lines = f.readlines()
        keyframe_indices = [int(line.strip()) for line in keyframe_lines]
        return keyframe_indices

    def get_pickupable_names(self) -> List[str]:
        return self.sssd_data.pickupables_in_scene

    def get_receptacles_names(self) -> List[str]:
        return self.sssd_data.receptacles_in_scene

    def get_assignment(self) -> dict:
        assignment = np.concatenate(
            [data._assignment for data in self.sssd_data.get_generator_of_selves()]
        )
        n_pickupables = assignment.shape[1]
        n_receptacles = assignment.shape[2]
        receptacle_names = self.get_receptacles_names()
        pickupable_names = self.get_pickupable_names()
        pickupable_assignment = dict()
        for p_id in range(n_pickupables):
            p_assignments = assignment[:, p_id, :]
            for r_id in range(n_receptacles):
                # Get positions where the pickupable is either present or absent in the receptacle
                mask = jnp.logical_or(
                    p_assignments[:, r_id] == 0, p_assignments[:, r_id] == 1
                )
                # Do not populate data if there are no valid timestamps
                if (
                    jnp.sum(mask) == 0
                    or receptacle_names[r_id] == "OOB_FAKE_RECEPTACLE"
                ):
                    continue
                r_assignment = p_assignments[:, r_id][mask]
                state = bool(jnp.median(r_assignment).item())
                pickupable_assignment[pickupable_names[p_id]] = state
                # If the pickupable is found, just return that it is present
                if state:
                    break

        return pickupable_assignment

    def get_pickupables_bbox(self) -> dict:
        oobb = self.sssd_data._oobb_cornerPoints[0]
        new_pickupables_bbox = {}
        for i, corners in enumerate(oobb):
            corners_hom = np.pad(corners, ((0, 0), (0, 1)), constant_values=1)
            corners_transformed = (RHS_TO_ROS @ LHS_TO_RHS @ corners_hom.T).T

            new_pickupables_bbox[self.get_pickupable_names()[i]] = {
                "cornerPoints": corners_transformed[:, :3]
            }

        return new_pickupables_bbox

    def get_receptacles_bbox(self) -> dict:
        new_receptacles_bbox = {}
        for object_name in self.get_receptacles_names():
            bbox = deepcopy(self.sssd_data.get_receptacle_oobb(object_name))

            # Process corners
            corners = np.array(bbox["cornerPoints"])
            corners_hom = np.pad(corners, ((0, 0), (0, 1)), constant_values=1)
            corners_transformed = (RHS_TO_ROS @ LHS_TO_RHS @ corners_hom.T).T
            bbox["cornerPoints"] = corners_transformed[:, :3]

            new_receptacles_bbox[object_name] = bbox

        return new_receptacles_bbox

    def get_pickupable_to_receptacles(self) -> dict:
        return self.sssd_data.pickupable_to_receptacle

    def get_timestamps(self) -> List[float]:
        timestamps = np.concatenate(
            [data._timestamp for data in self.sssd_data.get_generator_of_selves()]
        )
        timestamps = timestamps[
            self.sequence_start : self.sequence_end : self.sequence_stride
        ]

        return timestamps

    def get_rgb_paths(self) -> List[str]:
        path_str = str(
            self.base_path / self.scene / self.img_dir / self.rgb_dir / "*.jpg"
        )
        rgb_paths = natsorted(glob.glob(path_str))
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        path_str = str(
            self.base_path / self.scene / self.img_dir / self.depth_dir / "*.png"
        )
        depth_paths = natsorted(glob.glob(path_str))
        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        pose_path = str(self.base_path / self.scene / self.pose_dir / "*.json")
        pose_paths = natsorted(glob.glob(pose_path))
        poses = []

        for path in pose_paths:
            with open(path, "r") as f:
                pose_data = json.load(f)

            position = pose_data["position"]
            rotation = pose_data["rotation"]

            # 1. Construct the Matrix in the ORIGINAL Unity Frame (Left-Handed)
            # Unity Rotation = Yaw (Global Y) * Pitch (Local X)
            yaw = rotation["y"]
            pitch = rotation["x"]

            # Create rotations.
            # Note on AI2-THOR/Unity:
            # Yaw rotates around the global UP (Y).
            # Pitch rotates around the local Right.
            r_yaw = R.from_euler("y", yaw, degrees=True).as_matrix()
            r_pitch = R.from_euler("x", pitch, degrees=True).as_matrix()

            # Combined rotation in Unity Frame
            rot_unity = r_yaw @ r_pitch

            # Full Unity Pose Matrix
            pose_unity = np.eye(4)
            pose_unity[0:3, 0:3] = rot_unity
            pose_unity[0:3, 3] = [
                position["x"],
                position["y"] + self.cam_offset,
                position["z"],
            ]

            # 2. Apply Change of Basis: P_new = T * P_old * T_inv
            # Note: we do not invert T because T is its own inverse
            pose_rhs = RHS_TO_ROS @ (LHS_TO_RHS @ pose_unity @ LHS_TO_RHS.T)

            poses.append(pose_rhs)

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
        semantics = self.read_semantics(self.semantics_paths[idx])
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
        if (
            semantics.shape[0] != self.resized_height
            or semantics.shape[1] != self.resized_width
        ):
            semantics = cv2.resize(
                semantics,
                (self.resized_width, self.resized_height),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.relative_pose:
            pose = np.dot(self.first_pose_inv, pose)

        result = dict(
            rgb=rgb,
            depth=depth,
            semantics=semantics,
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
