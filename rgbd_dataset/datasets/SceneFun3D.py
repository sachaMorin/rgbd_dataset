import glob
import numpy as np
from typing import List
from pathlib import Path
from ..BaseRGBDDataset import BaseRGBDDataset
import os
import cv2

data_asset_to_path = {
    "lowres_wide": "<data_dir>/<visit_id>/<video_id>/lowres_wide/",
    "lowres_wide_intrinsics": "<data_dir>/<visit_id>/<video_id>/lowres_wide_intrinsics/",
    "lowres_depth": "<data_dir>/<visit_id>/<video_id>/lowres_depth/",
    "confidence": "<data_dir>/<visit_id>/<video_id>/confidence/",
    "hires_wide": "<data_dir>/<visit_id>/<video_id>/hires_wide/",
    "hires_wide_intrinsics": "<data_dir>/<visit_id>/<video_id>/hires_wide_intrinsics/",
    "hires_depth": "<data_dir>/<visit_id>/<video_id>/hires_depth/",
    # 'vga_wide',
    # 'vga_wide_intrinsics',
    # 'ultrawide',
    # 'ultrawide_intrinsics',
    "lowres_poses": "<data_dir>/<visit_id>/<video_id>/lowres_poses.traj",
    "hires_poses": "<data_dir>/<visit_id>/<video_id>/hires_poses.traj",
    "vid_mov": "<data_dir>/<visit_id>/<video_id>/<video_id>.mov",
    "vid_mp4": "<data_dir>/<visit_id>/<video_id>/<video_id>.mp4",
    "arkit_mesh": "<data_dir>/<visit_id>/<video_id>/<video_id>_arkit_mesh.ply",
    # '3dod_annotation',
    "laser_scan_5mm": "<data_dir>/<visit_id>/<visit_id>_laser_scan.ply",
    "crop_mask": "<data_dir>/<visit_id>/<visit_id>_crop_mask.npy",
    "transform": "<data_dir>/<visit_id>/<video_id>/<video_id>_transform.npy",
    "annotations": "<data_dir>/<visit_id>/<visit_id>_annotations.json",
    "descriptions": "<data_dir>/<visit_id>/<visit_id>_descriptions.json",
    "motions": "<data_dir>/<visit_id>/<visit_id>_motions.json",
}


def convert_angle_axis_to_matrix3(angle_axis):
    """
    Converts a rotation from angle-axis representation to a 3x3 rotation matrix.

    Args:
        angle_axis (numpy.ndarray): A 3-element array representing the rotation in angle-axis form.

    Returns:
        (numpy.ndarray): A 3x3 rotation matrix representing the same rotation as the input angle-axis.

    Raises:
        ValueError: If the input is not a valid 3-element numpy array.
    """
    # Check if input is a numpy array
    if not isinstance(angle_axis, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    # Check if the input is of shape (3,)
    if angle_axis.shape != (3,):
        raise ValueError(
            "Input must be a 3-element array representing the rotation in angle-axis representation."
        )

    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix


class SceneFun3D(BaseRGBDDataset):
    def __init__(self, base_path, scene, **kwargs):
        self.data_root_path = os.path.join(base_path)
        self.visit_id = scene

        self.scene_path = Path(self.data_root_path) / self.visit_id
        self.video_ids = [d.name for d in self.scene_path.iterdir() if d.is_dir()]
        self.common_timestamps = self.get_common_timestamps()

        print(f"Number of videos: {len(self.video_ids)}")

        super().__init__(base_path=base_path, scene=scene, **kwargs)

    def get_common_timestamps(self) -> List[List[str]]:
        """
        Finds timestamps common across RGB, depth, intrinsics, and poses for each video_id separately.

        Returns:
            List[List[str]]: A list where each element is a sorted list of timestamps common to all data types in a video_id.
        """
        common_timestamps_per_video = []

        for video_id in self.video_ids:
            rgb = self.scenefun3d_get_rgb_frames(self.visit_id, video_id)
            depth = self.scenefun3d_get_depth_frames(self.visit_id, video_id)
            intrinsics = self.scenefun3d_get_camera_intrinsics(self.visit_id, video_id)
            poses = self.scenefun3d_get_camera_trajectory(self.visit_id, video_id)

            video_common_timestamps = sorted(
                set(rgb.keys())
                & set(depth.keys())
                & set(intrinsics.keys())
                & set(poses.keys())
            )
            common_timestamps_per_video.append(list(video_common_timestamps))

        return common_timestamps_per_video

    def get_rgb_paths(self) -> List[str]:
        rgb_paths = []

        for idx, video_id in enumerate(self.video_ids):
            video_common_timestamps = self.common_timestamps[idx]
            rgb_dict = self.scenefun3d_get_rgb_frames(
                visit_id=self.visit_id, video_id=video_id
            )
            rgb_paths.extend(rgb_dict[ts] for ts in video_common_timestamps)

        print(f"Number of frames: {len(rgb_paths)}")
        return rgb_paths

    def get_depth_paths(self) -> List[str]:
        depth_paths = []

        for idx, video_id in enumerate(self.video_ids):
            video_common_timestamps = self.common_timestamps[idx]
            depth_dict = self.scenefun3d_get_depth_frames(
                visit_id=self.visit_id, video_id=video_id
            )
            depth_paths.extend(depth_dict[ts] for ts in video_common_timestamps)

        return depth_paths

    def get_se3_poses(self) -> List[np.array]:
        se3_poses = []

        for idx, video_id in enumerate(self.video_ids):
            video_common_timestamps = self.common_timestamps[idx]
            poses_dict = self.scenefun3d_get_camera_trajectory(
                visit_id=self.visit_id, video_id=video_id
            )
            se3_poses.extend(poses_dict[ts] for ts in video_common_timestamps)

        return se3_poses

    def get_intrinsic_matrices(self) -> List[np.array]:
        intrinsic_matrices = []

        for idx, video_id in enumerate(self.video_ids):
            video_common_timestamps = self.common_timestamps[idx]
            intrinsics_dict = self.scenefun3d_get_camera_intrinsics(
                visit_id=self.visit_id, video_id=video_id
            )
            intrinsic_matrices.extend(
                self.scenefun3d_read_camera_intrinsics(
                    intrinsics_dict[ts], format="matrix"
                )
                for ts in video_common_timestamps
            )

        return intrinsic_matrices

    def TrajStringToMatrix(self, traj_str):
        """
        Converts a line from the camera trajectory file into translation and rotation matrices.

        Args:
            traj_str (str): A space-delimited string where each line represents a camera pose at a particular timestamp.
                            The line consists of seven columns:
                - Column 1: timestamp
                - Columns 2-4: rotation (axis-angle representation in radians)
                - Columns 5-7: translation (in meters)

        Returns:
            (tuple): A tuple containing:
                - ts (str): Timestamp.
                - Rt (numpy.ndarray): 4x4 transformation matrix representing rotation and translation.

        Raises:
            AssertionError: If the input string does not have exactly seven columns.
        """
        tokens = traj_str.split()
        assert len(tokens) == 7
        ts = tokens[0]

        # Rotation in angle axis
        angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))

        # Translation
        t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
        extrinsics = np.eye(4, 4)
        extrinsics[:3, :3] = r_w_to_p
        extrinsics[:3, -1] = t_w_to_p
        Rt = np.linalg.inv(extrinsics)

        return (ts, Rt)

    def scenefun3d_get_camera_trajectory(
        self, visit_id, video_id, pose_source="colmap"
    ):
        """
        Retrieve the camera trajectory from a file and convert it into a dictionary whose keys are timestamps and
        values are the corresponding camera poses.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.
            pose_source (str, optional): Specifies the trajectory asset type, either "colmap" or "arkit". Defaults to "colmap".

        Returns:
            (dict): A dictionary where keys are timestamps (rounded to 3 decimal points) and values are 4x4 transformation matrices representing camera poses.

        Raises:
            AssertionError: If an unsupported trajectory asset type is provided.
        """
        assert pose_source in ["colmap", "arkit"], f"Unknown option {pose_source}"

        data_asset_identifier = (
            "hires_poses" if pose_source == "colmap" else "lowres_poses"
        )
        traj_file_path = self.get_data_asset_path(
            data_asset_identifier=f"{data_asset_identifier}",
            visit_id=visit_id,
            video_id=video_id,
        )

        with open(traj_file_path) as f:
            traj = f.readlines()

        # Convert trajectory to a dictionary
        poses_from_traj = {}
        for line in traj:
            traj_timestamp = line.split(" ")[0]

            if pose_source == "colmap":
                poses_from_traj[f"{float(traj_timestamp)}"] = np.array(
                    self.TrajStringToMatrix(line)[1].tolist()
                )
            elif pose_source == "arkit":
                poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = np.array(
                    self.TrajStringToMatrix(line)[1].tolist()
                )

        return poses_from_traj

    def scenefun3d_get_rgb_frames(
        self, visit_id, video_id, data_asset_identifier="hires_wide"
    ):
        """
        Retrieve the paths to the RGB frames for a given scene and video sequence.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.
            data_asset_identifier (str, optional): The data asset type for the RGB frames.
                                                   Can be either "hires_wide" or "lowres_wide".
                                                   Defaults to "hires_wide".

        Returns:
            (dict): A dictionary mapping frame timestamps to their corresponding file paths.

        Raises:
            ValueError: If an unsupported data asset identifier is provided.
            FileNotFoundError: If no frames are found at the specified path.
        """
        frame_mapping = {}
        if data_asset_identifier == "hires_wide":
            rgb_frames_path = self.get_data_asset_path(
                data_asset_identifier="hires_wide", visit_id=visit_id, video_id=video_id
            )

            frames = sorted(glob.glob(os.path.join(rgb_frames_path, "*.jpg")))
            if not frames:
                raise FileNotFoundError(f"No RGB frames found in {rgb_frames_path}")
            frame_timestamps = [
                os.path.basename(x).split(".jpg")[0].split("_")[1] for x in frames
            ]

        elif data_asset_identifier == "lowres_wide":
            rgb_frames_path = self.get_data_asset_path(
                data_asset_identifier="lowres_wide",
                visit_id=visit_id,
                video_id=video_id,
            )

            frames = sorted(glob.glob(os.path.join(rgb_frames_path, "*.png")))
            if not frames:
                raise FileNotFoundError(f"No RGB frames found in {rgb_frames_path}")
            frame_timestamps = [
                os.path.basename(x).split(".png")[0].split("_")[1] for x in frames
            ]
        else:
            raise ValueError(
                f"Unknown data_asset_identifier {data_asset_identifier} for RGB frames"
            )

        # Create mapping from timestamp to full path
        frame_mapping = {
            timestamp: frame for timestamp, frame in zip(frame_timestamps, frames)
        }

        return frame_mapping

    def scenefun3d_get_depth_frames(
        self, visit_id, video_id, data_asset_identifier="hires_depth"
    ):
        """
        Retrieve the paths to the depth frames for a given scene and video sequence.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.
            data_asset_identifier (str, optional): The data asset type for the depth frames.
                                                   Can be either "hires_depth" or "lowres_depth".
                                                   Defaults to "hires_depth".

        Returns:
            (dict): A dictionary mapping frame timestamps to their corresponding file paths.

        Raises:
            ValueError: If an unsupported data asset identifier is provided.
            FileNotFoundError: If no depth frames are found at the specified path.
        """
        frame_mapping = {}
        if data_asset_identifier == "hires_depth":
            depth_frames_path = self.get_data_asset_path(
                data_asset_identifier="hires_depth",
                visit_id=visit_id,
                video_id=video_id,
            )

        elif data_asset_identifier == "lowres_depth":
            depth_frames_path = self.get_data_asset_path(
                data_asset_identifier="lowres_depth",
                visit_id=visit_id,
                video_id=video_id,
            )

        else:
            raise ValueError(
                f"Unknown data_asset_identifier {data_asset_identifier} for depth frames"
            )

        frames = sorted(glob.glob(os.path.join(depth_frames_path, "*.png")))
        if not frames:
            raise FileNotFoundError(f"No depth frames found in {depth_frames_path}")
        frame_timestamps = [
            os.path.basename(x).split(".png")[0].split("_")[1] for x in frames
        ]

        # Create mapping from timestamp to full path
        frame_mapping = {
            timestamp: frame for timestamp, frame in zip(frame_timestamps, frames)
        }

        return frame_mapping

    def scenefun3d_get_camera_intrinsics(
        self, visit_id, video_id, data_asset_identifier="hires_wide_intrinsics"
    ):
        """
        Retrieve the camera intrinsics for a given scene and video sequence.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.
            data_asset_identifier (str, optional): The data asset type for camera intrinsics.
                                                   Can be either "hires_wide_intrinsics" or "lowres_wide_intrinsics".
                                                   Defaults to "hires_wide_intrinsics".

        Returns:
            (dict): A dictionary mapping timestamps to file paths of camera intrinsics data.

        Raises:
            ValueError: If an unsupported data asset identifier is provided.
            FileNotFoundError: If no intrinsics files are found at the specified path.
        """
        intrinsics_mapping = {}
        if data_asset_identifier == "hires_wide_intrinsics":
            intrinsics_path = self.get_data_asset_path(
                data_asset_identifier="hires_wide_intrinsics",
                visit_id=visit_id,
                video_id=video_id,
            )

        elif data_asset_identifier == "lowres_wide_intrinsics":
            intrinsics_path = self.get_data_asset_path(
                data_asset_identifier="lowres_wide_intrinsics",
                visit_id=visit_id,
                video_id=video_id,
            )

        else:
            raise ValueError(
                f"Unknown data_asset_identifier {data_asset_identifier} for camera intrinsics"
            )

        intrinsics = sorted(glob.glob(os.path.join(intrinsics_path, "*.pincam")))

        if not intrinsics:
            raise FileNotFoundError(f"No camera intrinsics found in {intrinsics_path}")

        intrinsics_timestamps = [
            os.path.basename(x).split(".pincam")[0].split("_")[1] for x in intrinsics
        ]

        # Create mapping from timestamp to full path
        intrinsics_mapping = {
            timestamp: cur_intrinsics
            for timestamp, cur_intrinsics in zip(intrinsics_timestamps, intrinsics)
        }

        return intrinsics_mapping

    def scenefun3d_read_camera_intrinsics(self, intrinsics_file_path, format="tuple"):
        """
        Parses a file containing camera intrinsic parameters and returns them in the specified format.

        Args:
            intrinsics_file_path (str): The path to the file containing camera intrinsic parameters.
            format (str, optional): The format in which to return the camera intrinsic parameters.
                                    Supported formats are "tuple" and "matrix". Defaults to "tuple".

        Returns:
            (Union[tuple, numpy.ndarray]): Camera intrinsic parameters in the specified format.

                - If format is "tuple", returns a tuple \\(w, h, fx, fy, hw, hh\\).
                - If format is "matrix", returns a 3x3 numpy array representing the camera matrix.

        Raises:
            ValueError: If an unsupported format is specified.
        """
        w, h, fx, fy, hw, hh = np.loadtxt(intrinsics_file_path)

        if format == "tuple":
            return (w, h, fx, fy, hw, hh)
        elif format == "matrix":
            return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])
        else:
            raise ValueError(f"Unknown format {format}")

    def get_data_asset_path(self, data_asset_identifier, visit_id, video_id=None):
        """
        Get the file path for a specified data asset.

        Args:
            data_asset_identifier (str): A string identifier for the data asset.
            visit_id (str or int): The identifier for the visit (scene).
            video_id (str or int, optional): The identifier for the video sequence. Required if specified data asset requires a video identifier.

        Returns:
            (Path): A Path object representing the file path to the specified data asset.

        Raises:
            AssertionError: If the `data_asset_identifier` is not valid or if `video_id` is required but not provided.
        """
        assert (
            data_asset_identifier in data_asset_to_path
        ), f"Data asset identifier '{data_asset_identifier}' is not valid"

        data_path = data_asset_to_path[data_asset_identifier]

        if ("<video_id>" in data_path) and (video_id is None):
            assert (
                False
            ), f"video_id must be specified for the data asset identifier '{data_asset_identifier}'"

        visit_id = str(visit_id)

        data_path = data_path.replace("<data_dir>", self.data_root_path).replace(
            "<visit_id>", visit_id
        )

        if "<video_id>" in data_path:
            video_id = str(video_id)
            data_path = data_path.replace("<video_id>", video_id)

        return data_path
