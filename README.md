# rgbd_dataset

A PyTorch `Dataset` for posed RGB-D sequences generated with
[SemiStaticSim](https://pypi.org/project/semistaticsim/). It is the data loader
behind Predictive Graphs: each item is one camera frame with its RGB image,
metric depth, semantic mask, camera pose, intrinsics, simulation timestamp and
(optionally) a world-frame Open3D point cloud. The dataset also exposes the
scene metadata used for evaluation: pickupable and receptacle names, their
oriented bounding boxes, and the ground-truth object-receptacle assignment.

## Installation

Requires Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
git clone <this repository> rgbd_dataset
cd rgbd_dataset
uv venv --python 3.12
uv pip install -e ".[test]"
```

or with pip inside an existing Python 3.12 environment: `pip install -e ".[test]"`.

On Linux this installs JAX with CUDA 12 support (pulled in by `semistaticsim`);
on macOS it installs the CPU build.

When used from Predictive Graphs, the package is a uv workspace member of that
repository and is installed by its `uv sync`; nothing else is needed.

## Data layout

`base_path` points at a SemiStaticSim `privileged` directory, and `scene` selects
one run inside it:

```
<layout>/0/
├── groundtruth/                  # used by get_assignment()
└── privileged/                   # <- base_path
    ├── run_0/                    # <- scene
    │   ├── images/
    │   │   ├── rgb_first_view/00000.jpg ...
    │   │   ├── depth/00000.png ...      # uint16, millimetres
    │   │   └── semantics/00000.png ...
    │   ├── poses/00000.json ...         # agent position + Euler angles (Unity frame)
    │   ├── keyframes.txt                # optional, one frame index per line
    │   └── scan_0.parquet, *.json       # SemiStaticSim scene metadata
    └── run_1/ ...
```

Poses are converted from Unity's left-handed, y-up frame to a right-handed
ROS frame (x forward, z up). `cam_offset` is added to the agent's height to get
the camera position.

## Usage

```python
from rgbd_dataset.datasets.SemiStaticSim import SemiStaticSim

dataset = SemiStaticSim(
    dataset_name="semistaticsim",
    base_path="/path/to/<layout>/0/privileged",
    scene="run_0",
    width=640, height=640,
    fx=320.0, fy=320.0, cx=320.0, cy=320.0,
    depth_scale=1000.0,
    sequence_stride=10,
)

obs = dataset[0]
obs["rgb"]          # (H, W, 3) uint8
obs["depth"]        # (H, W) float, metres
obs["semantics"]    # (H, W, 3) uint8 instance colours
obs["camera_pose"]  # (4, 4) camera-to-world
obs["intrinsics"]   # (3, 3), rescaled if resized_width/height are set
obs["timestamp"]    # simulation time of the frame
obs["point_cloud"]  # open3d.geometry.PointCloud in world frame (point_cloud=True)

dataset.get_pickupable_names()           # list[str]
dataset.get_receptacles_names()          # list[str]
dataset.get_pickupable_to_receptacles()  # dict[str, list[str]]
dataset.get_receptacles_bbox()           # {name: {"cornerPoints": (8, 3)}, ...}
dataset.get_pickupables_bbox()           # {name: {"cornerPoints": (8, 3)}, ...}
dataset.get_assignment()                 # {pickupable: present in the scene?}
```

The same dataset can be built from the Hydra config in
`src/rgbd_dataset/conf/dataset/SemiStaticSim.yaml` with `hydra.utils.instantiate`.

### Parameters

All of these can be overridden from the command line, for example `dataset.sequence_stride=20`.

| Parameter | Meaning |
| --- | --- |
| `base_path`, `scene` | Data is read from `base_path/scene`. |
| `width`, `height` | Original image size. |
| `resized_width`, `resized_height` | Output size; `-1` keeps the original. Intrinsics are rescaled to match. |
| `sequence_start`, `sequence_end`, `sequence_stride` | Frame slice; `sequence_end=-1` loads to the last frame. |
| `keyframes` | Load only the frames listed in `keyframes.txt` (ignores `sequence_stride`). |
| `relative_pose` | Express poses relative to the first frame. |
| `depth_scale` | Raw depth units per metre (`1000.0` for millimetres). |
| `point_cloud`, `depth_trunc` | Also return a world-frame point cloud, truncated at `depth_trunc` metres. |
| `fx`, `fy`, `cx`, `cy` | Pinhole intrinsics shared by all frames. |
| `cam_offset` | Camera height above the agent position, in metres. |

## Visualising a run

`rgbd-viz` (installed with the package) fuses every frame into one point cloud
and serves it, with the camera frustums and the receptacle bounding boxes, in a
[viser](https://viser.studio) viewer:

```bash
rgbd-viz dataset.base_path=/path/to/<layout>/0/privileged dataset.scene=run_0 dataset.sequence_stride=20
```

Open http://localhost:8080 and use the side panel to toggle frames, boxes and the
origin; stop it with Ctrl+C. `python -m rgbd_dataset.pcd_scene` is equivalent.
Add `--cfg job` to print the resolved config without loading any data.

## Tests

```bash
pytest                                                          # geometry and import tests
SEMISTATICSIM_DATA=/path/to/<layout>/0/privileged pytest        # also run the dataset and viewer tests
```

Tests that need data are skipped when `SEMISTATICSIM_DATA` is not set.

## Acknowledgements

Parts of the dataset interface follow the data utilities of
[gradslam](https://github.com/gradslam/gradslam).

## License

MIT, see [LICENSE](LICENSE).
