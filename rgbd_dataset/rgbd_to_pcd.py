import numpy as np
import open3d as o3d
import cv2


def rgbd_to_pcd(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_pose: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    depth_trunc: float = 8.0,
    depth_scale: float = 1.0,
) -> o3d.geometry.PointCloud:
    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)
    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        convert_rgb_to_intensity=False,
        depth_trunc=depth_trunc,
        depth_scale=depth_scale,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd_image,
        intrinsic=intrinsics_o3d,
        extrinsic=np.linalg.inv(camera_pose),  # World to cam
    )

    return pcd
