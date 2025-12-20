import json

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import os
from .RGBD import RGBD
from typing import List
import cv2
from tqdm import tqdm
from PIL import Image
import shutil

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1440
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 4.5  # noqa unused, left as a note to the user


class StrayScannerRGBD(RGBD):
    def __init__(
            self,
            rgb_video_file: str = "rgb.mp4",
            pose_file: str = "odometry.csv",
            intrinsics_file: str = "camera_matrix.csv",
            strayscanner_depth_confidence: int = 1,  # 0,1,2
            rotate: int = 180,  # 90, 180, or 270 degrees
            **kwargs,
    ):
        """
        StrayScanner dataset loader (inheriting from RGBD but reading its own format).
        """
        self.pose_file = pose_file
        self.intrinsics_file = intrinsics_file
        self.rgb_video_file = rgb_video_file
        self.strayscanner_depth_confidence = strayscanner_depth_confidence
        assert rotate in [0, 90, 180, 270, -90], "Rotation must be 0, -90, 90, 180, or 270 degrees"
        if rotate == -90:
            rotate = 270
        if rotate in [90,270]:
            raise NotImplementedError("Weird behaviour; fixme")
        self.rotate = rotate
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def get_cache_sentinel(self):
        return {
            "rgb_video_file": self.rgb_video_file,
            "pose_file": self.pose_file,
            "intrinsics_file": self.intrinsics_file,
            "strayscanner_depth_confidence": self.strayscanner_depth_confidence,
            "rotate": self.rotate,
            **self.kwargs,
        }

    def save_cache_sentinel(self, path):
        path = path / "SENTINEL.json"
        with open(path, "w") as f:
            json.dump(self.get_cache_sentinel(), f)

    def read_cache_sentinel(self, path):
        path = path / "SENTINEL.json"
        with open(path, "r") as f:
            return json.load(f)

    def cache_sentinel_equality(self, path):
        try:
            read_sentinel = self.read_cache_sentinel(path)
        except FileNotFoundError:
            return False

        for k, v in self.get_cache_sentinel().items():
            if k not in read_sentinel:
                return False
            if read_sentinel[k] != v:
                return False
        return True

    def get_depth_paths(self) -> List[str]:
        DEPTH_PATH = Path(self.base_path) / self.scene / self.depth_dir
        CONFIDENCE_PATH = Path(self.base_path) / self.scene / "confidence"
        RAW_DEPTH_PATH = Path(self.base_path) / self.scene / "raw_depth"

        if self.cache_sentinel_equality(DEPTH_PATH):
            return super().get_depth_paths()

        if not os.path.exists(str(RAW_DEPTH_PATH)):
            shutil.copytree(str(DEPTH_PATH), RAW_DEPTH_PATH, dirs_exist_ok=True)
        if os.path.exists(DEPTH_PATH):
            shutil.rmtree(str(DEPTH_PATH))
        os.makedirs(str(DEPTH_PATH), exist_ok=True)

        def resize_depth(depth):
            out = cv2.resize(depth, (self.resized_width, self.resized_height), interpolation=cv2.INTER_NEAREST_EXACT)
            out[out < 10] = 0
            return out

        files = list(sorted(os.listdir(RAW_DEPTH_PATH)))
        for filename in tqdm(files, desc="Processing depth frames"):
            if not ('.npy' in filename or '.png' in filename):
                continue
            number, _ = filename.split('.')
            confidence = np.array(Image.open(os.path.join(CONFIDENCE_PATH, number + '.png')))
            depth = np.array(Image.open(os.path.join(RAW_DEPTH_PATH, filename)))
            depth[confidence < self.strayscanner_depth_confidence] = 0
            depth = resize_depth(depth)
            if self.rotate == 90:
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotate == 180:
                depth = cv2.rotate(depth, cv2.ROTATE_180)
            elif self.rotate == 270:
                depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(DEPTH_PATH, f'{number}.png'), depth)

        self.save_cache_sentinel(DEPTH_PATH)
        return super().get_depth_paths()

    def get_rgb_paths(self) -> List[str]:
        RGB_PATH = Path(self.base_path) / self.scene / self.rgb_dir
        RGB_VIDEO_PATH = Path(self.base_path) / self.scene / self.rgb_video_file

        assert RGB_VIDEO_PATH.exists()
        os.makedirs(RGB_PATH, exist_ok=True)

        if self.cache_sentinel_equality(RGB_PATH):
            return super().get_rgb_paths()

        cap = cv2.VideoCapture(str(RGB_VIDEO_PATH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {RGB_VIDEO_PATH}")

        try:
            for i in tqdm(range(total_frames), desc="Writing rgb frames"):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (self.resized_width, self.resized_height))
                if self.rotate == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotate == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotate == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame_path = os.path.join(str(RGB_PATH), f"{i:06}.jpg")
                params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                cv2.imwrite(frame_path, frame, params)
        finally:
            cap.release()

        self.save_cache_sentinel(RGB_PATH)
        return super().get_rgb_paths()

    def get_intrinsic_matrices(self):
        """
        Loads the intrinsics from StrayScanner's camera_matrix.csv.
        The file format is:
        fx, 0, cx
        0, fy, cy
        (no 3rd row; we append [0, 0, 1])
        """
        INTRINSICS_PATH = Path(self.base_path) / self.scene / self.intrinsics_file
        assert INTRINSICS_PATH.exists()

        matrix = np.loadtxt(INTRINSICS_PATH, delimiter=',')
        fx = matrix[0, 0]
        fy = matrix[1, 1]
        cx = matrix[0, 2]
        cy = matrix[1, 2]
        matrix = np.array([[fx, 0.0, cx],
                           [0., fy, cy],
                           [0., 0., 1.0]])

        if self.rotate == 180:
            matrix[0, 2] = FRAME_WIDTH - matrix[0, 2]  # new cx
            matrix[1, 2] = FRAME_HEIGHT - matrix[1, 2]  # new cy
        elif self.rotate == 90:
            # Swap fx/fy and cx/cy after rotation
            matrix = np.array([
                [fy, 0, cy],
                [0, fx, FRAME_WIDTH - cx],
                [0, 0, 1]
            ])
        elif self.rotate == 270:
            matrix = np.array([
                [fy, 0, FRAME_HEIGHT - cy],
                [0, fx, cx],
                [0, 0, 1]
            ])

        # StrayScanner has 1 set of intrinsics; RGBD expects one per frame
        n_frames = len(self.get_rgb_paths())
        return [matrix for _ in range(n_frames)]

    def get_se3_poses(self):
        """
        Loads poses from StrayScanner's odometry.csv and converts to 4x4 SE(3) matrices.
        The CSV format is:
        timestamp, frame, x, y, z, qx, qy, qz, qw
        """
        POSE_PATH = Path(self.base_path) / self.scene / self.pose_file
        assert POSE_PATH.exists()

        odometry = np.loadtxt(POSE_PATH, delimiter=',', skiprows=1)
        poses = []
        for line in odometry:
            # timestamp, frame, x, y, z, qx, qy, qz, qw
            position = line[2:5]
            quaternion = line[5:]
            T_WC = np.eye(4)
            rot_matrix = R.from_quat(quaternion).as_matrix()

            def get_rot_mat(angle):
                R_z = R.from_euler('z', angle).as_matrix()
                return R_z

            # Apply rotation based on the selected degree
            if self.rotate == 90:
                rot_matrix = rot_matrix @ get_rot_mat(np.pi / 2)
            elif self.rotate == 180:
                rot_matrix = rot_matrix @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            elif self.rotate == 270:
                rot_matrix = rot_matrix @ get_rot_mat(-np.pi / 2)

            T_WC[:3, :3] = rot_matrix
            T_WC[:3, 3] = position
            poses.append(T_WC)
        return poses