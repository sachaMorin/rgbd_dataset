import glob
import numpy as np
from typing import List, Tuple
from natsort import natsorted
import json
from scipy.spatial.transform import Rotation as R

from ..BaseRGBDDataset import BaseRGBDDataset


class RGBD(BaseRGBDDataset):
    def __init__(
        self,
        rgb_dir: str = "rgb",
        depth_dir: str = "depth",
        pose_dir: str = "camera_pose",
        intrinsics_dir: str = "intrinsics",
        pose_dict: bool = False,
        **kwargs,
    ):
        self.rgb_dir = rgb_dir
        self.pose_dir = pose_dir
        self.depth_dir = depth_dir
        self.intrinsics_dir = intrinsics_dir
        self.pose_dict = pose_dict  # Pose is in dict format from ROS transform. Otherwise assumees a 4x4 matrix
        super().__init__(**kwargs)

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
            if self.pose_dict:
                pose_mx = np.eye(4)
                t = pose["translation"]
                pose_mx[:3, 3] = [t["x"], t["y"], t["z"]]
                rot = pose["rotation"]
                pose_mx[:3, :3] = R.from_quat(
                    (rot["x"], rot["y"], rot["z"], rot["w"]), scalar_first=False
                ).as_matrix()
            else:
                pose_mx = np.array(pose).reshape((4, 4))

            poses.append(pose_mx)
        return poses

    def get_intrinsic_matrices(self) -> List[np.array]:
        intrinsic_path = str(
            self.base_path / str(self.scene) / self.intrinsics_dir / "*.json"
        )
        paths = natsorted(glob.glob(intrinsic_path))
        intrinsics = []
        for path in paths:
            mx = json.loads(open(path).read())
            intrinsics.append(np.array(mx).reshape((3, 3)))
        return intrinsics

    def get_descriptions(self, use_test: bool = False) -> List[Tuple[str, str]]:
        if use_test:
            return self.get_descriptions_test()

        path_str = str(self.base_path / self.scene / "descriptions.json")

        # if the descriptions file doesn't exist, return an empty list
        if not (self.base_path / self.scene / "descriptions.json").exists():
            print(
                f"Warning: descriptions.json not found for scene {self.scene}. Returning empty list."
            )
            return []

        with open(path_str, "r") as f:
            descriptions = json.load(f)

        description_ids = [desc["desc_id"] for desc in descriptions]
        description_texts = [desc["description"] for desc in descriptions]
        return list(zip(description_ids, description_texts))

    def get_descriptions_test(self) -> List[Tuple[str, str]]:
        path_str = str(self.base_path / self.scene / "descriptions.json")

        if not (self.base_path / self.scene / "descriptions.json").exists():
            print(
                f"Warning: descriptions.json not found for scene {self.scene}. Returning empty list."
            )
            return []

        with open(path_str, "r") as f:
            descriptions = json.load(f)

        description_ids = [desc["desc_id"] for desc in descriptions]
        description_texts = [desc["description"] for desc in descriptions]
        return list(zip(description_ids, description_texts))
