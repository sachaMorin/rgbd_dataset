import open3d as o3d
from typing import List

from scipy.spatial.transform import Rotation as Rsc
from rgbd_dataset.utils import invert_se3
import numpy as np
import viser


class ViserServer:
    def __init__(self, point_shape: str = "circle"):
        self.server = viser.ViserServer()
        # Containers
        self.point_cloud: o3d.geometry.PointCloud
        self.point_cloud_names: List[str] = [] 
        self.point_cloud_handle: List[viser.PointCloudHandle] = []
        self.camera_poses: List[np.ndarray] = []
        self.camera_imgs: List[np.ndarray] = []
        self.frames: List[viser.FrameHandle] = []
        self.origin = viser.FrameHandle = None
        self.oobb: List[o3d.geometry.OrientedBoundingBox] = []
        self.box_handles: List[viser.BoxHandle] = []
        self.intrinsics: np.ndarray = None
        # Hyperparameters
        self.point_shape = point_shape
        self.downsample_factor = 2

        # GUI
        with self.server.gui.add_folder("Point Cloud"):
            self.pcd_size_gui_slider = self.server.gui.add_slider(
                "Point size",
                min=0.001,
                max=0.030,
                step=0.001,
                initial_value=0.010,
                disabled=False,
            )
            self.pcd_origin_gui_checkbox = self.server.gui.add_checkbox(
                "Show Origin",
                initial_value=False,
            )
            self.pcd_size_gui_slider.on_update(self.on_point_size_change)
            self.pcd_origin_gui_checkbox.on_update(self.on_show_origin_change)

        with self.server.gui.add_folder("Camera Frames"):
            self.frames_gui_checkbox = self.server.gui.add_checkbox(
                "Show Frames",
                initial_value=False,
            )
            self.frames_gui_counter = self.server.gui.add_number(
                "Number of Frames",
                initial_value=50,
                min=1)
            self.frames_gui_checkbox.on_update(self.on_show_frames_change)
            self.frames_gui_counter.on_update(self.on_show_frames_change)

        with self.server.gui.add_folder("Bounding Boxes"):
            self.boxes_gui_checkbox = self.server.gui.add_checkbox(
                "Show Boxes",
                initial_value=False,
            )
            self.boxes_gui_checkbox.on_update(self.on_show_boxes_change)

    def collate(self, point_cloud: o3d.geometry.PointCloud):
        # Downsample point clouds to make life easier for viser
        self.point_cloud = point_cloud.voxel_down_sample(voxel_size=0.005)

    def on_point_size_change(self, data):
        point_size = self.pcd_size_gui_slider.value

        for handle in self.point_cloud_handles:
            handle.point_size = point_size

    def on_show_frames_change(self, data):
        show_frames = self.frames_gui_checkbox.value
        if show_frames:
            self.display_camera_poses()
        else:
            self.clear_frames()

    def on_show_origin_change(self, data):
        show_origin = self.pcd_origin_gui_checkbox.value
        if show_origin:
            self.display_origin()
        else:
            self.origin.remove()
            self.origin = None

    def on_show_boxes_change(self, data):
        show_boxes = self.boxes_gui_checkbox.value
        if show_boxes:
            self.display_boxes()
        else:
            self.clear_boxes()

    def reset(self):
        self.server.scene.reset()
        self.display_object_rgb()

    def clear_scene(self):
        for name in self.point_cloud_names:
            self.server.scene.remove_by_name(name)
        self.point_cloud_names = []
        self.point_cloud_handles = []

    def clear_frames(self):
        for frame in self.frames:
            frame.remove()
        self.frames.clear()

    def clear_boxes(self):
        for box in self.box_handles:
            box.remove()
        self.box_handles = []

    def display_point_cloud(self):
        self.clear_scene()

        name = f"pcd_0"
        pcd_points = np.asarray(self.point_cloud.points)
        pcd_colors = np.asarray(self.point_cloud.colors)
        handle = self.server.add_point_cloud(
            name,
            pcd_points,
            pcd_colors,
            point_size=self.pcd_size_gui_slider.value,
            point_shape=self.point_shape,
        )
        self.point_cloud_names.append(name)
        self.point_cloud_handles.append(handle)

    def display_camera_poses(self):
        # Remove existing image frames.
        self.clear_frames()

        n_frames = len(self.camera_poses)
        n_frames_to_display = min(n_frames, int(self.frames_gui_counter.value))
        # Select the frames we will display
        indices = np.random.choice(n_frames, n_frames_to_display, replace=False)

        # Get focal length
        fy = self.intrinsics[1, 1]
        for i in indices:
            cam_pose = self.camera_poses[i]
            img = self.camera_imgs[i]
            cam_name = f"/frame_{i}"
            quat = Rsc.from_matrix(cam_pose[:3, :3]).as_quat(scalar_first=True)
            position = cam_pose[:3, 3]
            frame = self.server.scene.add_frame(
                name=cam_name,
                wxyz=quat,
                position=position,
                axes_length=0.1,
                axes_radius=0.005,
            )
            H, W = img.shape[:2]
            downsampled_img = img[::self.downsample_factor, ::self.downsample_factor]
            frustum = self.server.scene.add_camera_frustum(
                f"/frame_{i}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.15,
                image=downsampled_img,
            )

            self.frames.append(frame)

            @frustum.on_click
            def _(_, frame=frame) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

    def display_origin(self):
        origin = self.server.scene.add_frame(
            "/origin",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )
        self.origin = origin

    def display_boxes(self):
        # Clear existing boxes
        self.clear_boxes()

        for i, box in enumerate(self.oobb):
            box_name = f"box_{i}"
            center = box.center
            extent = box.extent
            R = np.array(box.R, copy=True)
            wxyz = Rsc.from_matrix(R).as_quat(scalar_first=True)

            box = self.server.scene.add_box(
                name=box_name,
                color=(0, 1, 0),
                dimensions=extent,
                position=center,
                wxyz=wxyz,
                visible=True,
                wireframe=True,
            )
            self.box_handles.append(box)

    def display_object_rgb(self):
        self.display_point_cloud()

    def add_oriented_bbox(self, object_index: int, label: str):
        # bbox around object + label
        bbox = self.object_map[object_index].pcd.get_oriented_bounding_box()
        center = bbox.center
        extent = bbox.extent
        R = np.array(bbox.R, copy=True)

        quat = Rsc.from_matrix(R).as_quat()  # x, y, z, w
        wxyz = (quat[3], quat[0], quat[1], quat[2])  # w, x, y, z

        name = f"bbox_{object_index}"
        self.server.scene.add_box(
            name=name,
            color=(0, 0, 0),
            dimensions=extent,
            position=center,
            wxyz=wxyz,
            visible=True,
            wireframe=True,
        )
        self.reasoning_annotation_names.append(name)

        if label != "":
            label_name = f"bbox_label_{object_index}"
            label_pos = center.copy()

            self.server.scene.add_label(
                name=label_name,
                text=label,
                position=label_pos,
                visible=True,
            )
            self.reasoning_annotation_names.append(label_name)