import dataclasses
import json
import os
import re
from typing import List, Optional, Tuple, Dict
from typing_extensions import Self
import yaml

import jax
import jax.random
import numpy as np
import polars as pl
from flax import struct
from jax import numpy as jnp
from pathlib import Path


def split_camel_preserve_acronyms(name):
    # Insert space between lowercase → uppercase
    # OR between acronym → normal word
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return s.lower()


def thin2wide(assignment, receptacles_in_scene):
    def do_for_each_p(r_or_n_value):
        mat = jnp.zeros(
            (
                len(
                    receptacles_in_scene,
                )
            )
        )
        mat = mat.at[r_or_n_value].set(1)
        return mat

    def for_each_timestep(p_r_apn):
        return jax.vmap(do_for_each_p)(p_r_apn)

    assignment = assignment.astype(int)
    return jax.vmap(for_each_timestep)(assignment)


def path_2_parts(path: str | Path) -> Tuple[str, int, jax.random.PRNGKey]:
    p = Path(path).resolve()
    parts = list(p.parts)
    for i in range(len(parts) - 2):
        if parts[i].isalpha() and parts[i + 1].isdigit() and parts[i + 2].isdigit():
            procthor_split = parts[i]
            procthor_index = int(parts[i + 1])
            jax_seed = int(parts[i + 2])
            return procthor_split, procthor_index, jax_seed


@struct.dataclass
class GeneratedSemiStaticData:
    split: str
    house_id: str
    jax_key: jax.random.PRNGKey  # seed
    dt: float

    _assignment: jnp.ndarray
    _timestamp: jnp.ndarray
    _aabb_center: jnp.ndarray
    _aabb_cornerPoints: jnp.ndarray
    _aabb_size: jnp.ndarray
    _oobb_cornerPoints: jnp.ndarray
    _position: jnp.ndarray
    _rotation: jnp.ndarray

    pickupable_names: List[str]
    receptacle_names: List[str]
    pickupable_to_receptacle: Dict[str, str]
    receptacles_oobb: Dict[str, Dict]

    is_single_timestamp: (
        bool  # Is self sliced at a timestamp (single timestamp strict)?
    )
    is_single_pickupable: bool  # Is self sliced at a pickupable (single pickupable strict and also single timestamp strict)?

    parquet_paths: List[str]

    intrinsics: Optional[Dict[str, Dict]] = None
    parquet_id: int = 0
    timestep: int = 0  # Which timestep index are we at
    parent: Self = None  # Self but with all info if sliced, otherwise none

    def __getitem__(self, item):
        """Index class attributes by string key"""
        if isinstance(item, str):
            return dataclasses.asdict(self)[item]
        raise NotImplementedError()

    def get_receptacle_oobb(self, receptacle_name) -> Dict[str, Dict]:
        """Get the receptacle OOBB information"""
        return self.receptacles_oobb[receptacle_name]

    @property
    def original_batch_size(self) -> int:
        """What is the parquets' batch size"""
        return self._resolve_original_parent()._timestamp.shape[0]

    @property
    def global_timestep(self) -> int:
        """What is the global timestep index across all parquets"""
        return self.original_batch_size * self.parquet_id + self.timestep

    @property
    def global_length(self):
        """Total number of timesteps across all parquets"""

        if len(self.parquet_paths) == 0:
            return self._timestamp.shape[0]

        temp_self = self.replace(
            parquet_id=len(self.parquet_paths) - 1
        ).resolve_parquet()
        return (
            len(self.parquet_paths) - 1
        ) * self.original_batch_size + temp_self._timestamp.shape[0]

    def _resolve_original_parent(self) -> Self:
        """Recursively resolve to the original parent if sliced, otherwise return self"""
        if self.parent is None:
            return self
        return self.parent._resolve_original_parent()

    def take_by_global_timestep(
        self, stop_or_start, stop=None, step=None, indices=None
    ) -> Self:
        """Index a slice of data across all parquets
        self.take_by_global_timestep(20, 50)  # Take data from index 20 (inclusive) to 50 (non inclusive) across all parquets
        self.take_by_global_timestep(None)    # Takes all data from all parquets
        self.take_by_global_timestep(None, indices=[0, 5, 10, 15])  # Take specific indices
        """
        if indices is not None:
            # Use the provided indices directly
            indices_to_take = jnp.asarray(indices).squeeze()
        else:
            # Use slice notation
            data_slice = slice(stop_or_start, stop, step)
            indices_to_take = jnp.arange(self.global_length)[data_slice].squeeze()

        parquet_id_to_timestep_id = {}
        og_batch_size = self.original_batch_size
        for i in indices_to_take:
            parquet_id = int(i / og_batch_size)
            timestep_id = int(i - parquet_id * og_batch_size)

            if parquet_id not in parquet_id_to_timestep_id:
                parquet_id_to_timestep_id[parquet_id] = []
            parquet_id_to_timestep_id[parquet_id].append(timestep_id)
        parquet_id_to_timestep_id = {
            k: jnp.array(v) for k, v in parquet_id_to_timestep_id.items()
        }

        ret = []
        for parquet_id, timestep_ids in parquet_id_to_timestep_id.items():
            self = self.replace(parquet_id=parquet_id, timestep=0).resolve_parquet()

            ret.append(
                self.replace(**{k: v[timestep_ids] for k, v in self.TimeVaryingItems()})
            )

        ret_self = ret[0]
        for self in ret[1:]:
            ret_self = ret_self.concat(self)
        return ret_self

    @property
    def pickupables_in_scene(self):
        """Get all pickupable names in the scene - same as pickupable_names.json"""
        return self._resolve_original_parent().pickupable_names

    @property
    def receptacles_in_scene(self):
        """Get all receptacle names in the scene - same as receptacle_names.json"""
        return self._resolve_original_parent().receptacle_names

    @classmethod
    def TimeVaryingKeys(cls) -> List[str]:
        """Returns the name of the keys that are in the parquet files (time varying)"""
        return [f.name for f in dataclasses.fields(cls) if f.name.startswith("_")]

    def TimeVaryingItems(self) -> dict.items:
        """Returns the items (key, value) of the time varying keys (parquets)"""
        asdict = dataclasses.asdict(self)
        return {k: v for k, v in asdict.items() if k.startswith("_")}.items()

    @property
    def is_full_timestamps(self) -> bool:
        """Check if current self is either a single timestamp or full data"""
        return not self.is_single_timestamp

    @property
    def is_full_pickupables(self) -> bool:
        """Check if current self is either a single pickupable or full data"""
        return not self.is_single_pickupable

    def unstep(self) -> Self:
        """
        Step to the next timestep, or next parquet if at the end of current parquet

        NOTE: Used mainly during data generation
        """
        self = self.replace(timestep=self.timestep - 1)
        if self.timestep < 0:
            if self.parquet_id > 0:
                self = self.replace(
                    parquet_id=self.parquet_id - 1,
                    timestep=self.original_batch_size - 1,
                )
                self = self.resolve_parquet()
            else:
                print("WARNING: all generated semi static object steps are DONE!")
                return None
        return self

    def step(self) -> Self:
        """
        Step to the next timestep, or next parquet if at the end of current parquet

        NOTE: Used mainly during data generation
        """
        self = self.replace(timestep=self.timestep + 1)
        if self.timestep >= self._assignment.shape[0]:
            if self.parquet_id < len(self.parquet_paths):
                self = self.replace(parquet_id=self.parquet_id + 1, timestep=0)
                self = self.resolve_parquet()
            else:
                print("WARNING: all generated semi static object steps are DONE!")
                return None
        return self

    def get_generator_of_selves(self):
        """Iterator that returns a (full) self for each parquet file"""
        yield self
        while self.parquet_id < len(self.parquet_paths) - 1:
            self = self.replace(parquet_id=self.parquet_id + 1, timestep=0)
            self = self.resolve_parquet()
            yield self

    def resolve_parquet(self) -> Self:
        """Update self by loading data from the parquet file at self.parquet_id - same as read_parquet"""
        ITEMS = {key: None for key in self.TimeVaryingKeys()}

        path = self.parquet_paths[self.parquet_id]
        df = pl.read_parquet(path)

        for key in self.TimeVaryingKeys():
            ITEMS[key] = jnp.array(df[key[1:]]).squeeze()

        if len(ITEMS["_assignment"].shape) == 2:
            # edge case: the parquet contains a single datapoint
            for key in self.TimeVaryingKeys():
                ITEMS[key] = ITEMS[key][None]

        # if len(ITEMS["_assignment"].shape) == 2:
        #    ITEMS["_assignment"] = thin2wide(ITEMS["_assignment"], self.receptacles_in_scene)

        self = self.replace(**ITEMS)
        return self

    @property
    def self_at_current_time(self) -> Self:
        """Slice self at current timestep index __get_item__(timestamp)
        where it retusn self as opposed to an element in the container"""
        return self.replace(
            **{k: v[self.timestep] for k, v in self.TimeVaryingItems()},
            is_single_timestamp=True,
            parent=self,
        )

    @property
    def pickupable_selves_at_current_time(self) -> List[Self]:
        """Returns a dictionary pickupable_name -> self sliced at current timestep and pickupable"""
        if not self.is_single_timestamp:
            current_time_self = self.self_at_current_time

        ret = {}
        for i, p in enumerate(self.pickupables_in_scene):
            new_self = current_time_self.replace(
                pickupable_names=[p],
                receptacle_names=self.pickupable_to_receptacle[p],
                **(
                    {
                        k: (v[i] if "timestamp" not in k else v)
                        for k, v in current_time_self.TimeVaryingItems()
                    }
                ),
                is_single_pickupable=True,
                parent=current_time_self,
            )
            ret[p] = new_self
        return ret

    def current_receptacle_for_this_pickupable(self, pickupable_name: str) -> str:
        """Which receptacle is the given pickupable currently in? If not returns unobserved message"""
        if self.is_single_pickupable:
            assert self.is_single_timestamp
            assignment = jnp.atleast_1d(self._assignment).argmax()
            return (
                self.receptacles_in_scene[assignment]
                if self._assignment[assignment] == 1
                else f"Pickupable {pickupable_name} was unobserved"
            )
        return self.pickupable_selves_at_current_time[
            pickupable_name
        ].current_receptacle_for_this_pickupable(pickupable_name)

    def is_this_pickupable_in_the_OOB_FAKE_RECEPTACLE(
        self, pickupable_name: str
    ) -> bool:
        """Check if a given pickupable is currently in the OOB_FAKE_RECEPTACLE"""
        return (
            self.current_receptacle_for_this_pickupable(pickupable_name)
            == "OOB_FAKE_RECEPTACLE"
        )

    def get_singletimestamp_prototype(self) -> Self:
        """
        Get a prototype self that is single timestamp with all data as NaN
        """
        kwargs = {}
        for k, v in self.self_at_current_time.TimeVaryingItems():
            if "assignment" in k:
                kwargs[k] = (
                    jnp.ones(
                        (len(self.pickupables_in_scene), len(self.receptacles_in_scene))
                    )
                    * jnp.nan
                )
            else:
                kwargs[k] = jnp.ones_like(v) * jnp.nan
        return self.replace(
            **kwargs,
            parent=None,
            is_single_timestamp=True,
            is_single_pickupable=False,
            pickupable_names=self.pickupables_in_scene,
            receptacle_names=self.receptacles_in_scene,
            parquet_paths=[],
            parquet_id=0,
        )

    def concat(self, other_self: Self):
        """Concatenate two selves along the timestamp dimension"""
        # assumes self to be the accumulator
        assert not self.is_single_pickupable

        if self.is_single_timestamp:
            # the current self is a point
            self = self.replace(
                **{k: v[None] for k, v in self.TimeVaryingItems()},
                is_single_timestamp=False,
            )

        kwargs = {}
        for k, v in self.TimeVaryingItems():
            if other_self.is_single_timestamp:
                kwargs[k] = jnp.concatenate([v, other_self[k][None]])
            else:
                kwargs[k] = jnp.concatenate([v, other_self[k]])

        concatenated_self = self.replace(**kwargs)
        return concatenated_self

    def remove_nan_sentinel(self):
        if jnp.isnan(self._timestamp[0]).any():
            self = self.replace(**{k: v[1:] for k, v in self.TimeVaryingItems()})
            assert not jnp.isnan(self._timestamp[0]).any()
        return self

    def backprop_majors(self):
        UNIQUE_TIMESTAMPS, UNIQUE_TIMESTAMP_INDEXES, UNIQUE_TIMESTAMP_COUNTS = (
            jnp.unique(self._timestamp, return_counts=True, return_index=True)
        )
        UNIQUE_TIMESTAMPS = list(map(float, UNIQUE_TIMESTAMPS))

        self_summed_majors = self.sum_up_minors_into_major(verbose=True)
        for i, (
            unique_timestamp,
            unique_timestamp_index,
            unique_timestamp_count,
        ) in enumerate(
            zip(UNIQUE_TIMESTAMPS, UNIQUE_TIMESTAMP_INDEXES, UNIQUE_TIMESTAMP_COUNTS)
        ):

            major_index = jnp.argmax(
                self_summed_majors._timestamp == int(unique_timestamp)
            )
            major_assignment = self_summed_majors._assignment[major_index]

            if i + 1 < len(UNIQUE_TIMESTAMP_INDEXES):
                slice = self._assignment.at[
                    UNIQUE_TIMESTAMP_INDEXES[i] : UNIQUE_TIMESTAMP_INDEXES[i + 1]
                ]
            else:
                slice = self._assignment.at[UNIQUE_TIMESTAMP_INDEXES[i] :]

            sliced = slice.get()
            masked = jnp.astype(
                sliced >= 0, int
            )  # indices all timestamps where we have non-missing data for the receptacles

            new_sliced = jax.vmap(lambda s, m: major_assignment * m + s * (1 - m))(
                sliced, masked
            )
            # new_sliced = major_assignment * masked + sliced * (1-masked)

            new_self = self.replace(_assignment=slice.set(new_sliced))
            assert new_self._assignment.shape == self._assignment.shape
            self = new_self
            del new_self

        return self

    def regress_minors_back_into_raw_minors(self):
        return self.replace(
            _timestamp=jnp.astype(jnp.astype(self._timestamp, int), float)
        )

    def sum_up_minors_into_major(self, verbose=True):
        if verbose:
            print(
                "Summing up minors into major. This only works if you call it right after a major loop has finished! Otherwise you might introduce bugs in the saved data."
            )

        assert not self.is_single_pickupable
        assert not self.is_single_timestamp

        self = self.regress_minors_back_into_raw_minors()

        UNIQUE_TIMESTAMPS, UNIQUE_TIMESTAMP_INDEXES, UNIQUE_TIMESTAMP_COUNTS = (
            jnp.unique(self._timestamp, return_counts=True, return_index=True)
        )
        UNIQUE_TIMESTAMPS = list(map(float, UNIQUE_TIMESTAMPS))

        PROTO = self.get_singletimestamp_prototype()
        MAJOR_TO_PROTO = {}
        for i, (value, index, count) in enumerate(
            zip(UNIQUE_TIMESTAMPS, UNIQUE_TIMESTAMP_INDEXES, UNIQUE_TIMESTAMP_COUNTS)
        ):
            if i + 1 < len(UNIQUE_TIMESTAMP_INDEXES):
                slice = self._assignment.at[
                    UNIQUE_TIMESTAMP_INDEXES[i] : UNIQUE_TIMESTAMP_INDEXES[i + 1]
                ]
            else:
                slice = self._assignment.at[UNIQUE_TIMESTAMP_INDEXES[i] :]
            slice = slice.get()

            MAJOR_FOR_Ps = []
            for p_id, p_name in enumerate(self.pickupables_in_scene):
                minor_assignments_for_p = slice[:, p_id, :]
                major_assignments_for_p = jnp.max(minor_assignments_for_p, axis=0)

                for r_id, r_name in enumerate(self.receptacles_in_scene):
                    if r_name not in self.pickupable_to_receptacle[p_name]:
                        major_assignments_for_p = major_assignments_for_p.at[r_id].set(
                            -2
                        )

                major_assignments_for_p = major_assignments_for_p[None, None, :]
                MAJOR_FOR_Ps.append(major_assignments_for_p)

            major = jnp.concatenate(MAJOR_FOR_Ps, axis=1)
            self_at_this_time = self.replace(
                timestep=UNIQUE_TIMESTAMP_INDEXES[i]
            ).self_at_current_time
            MAJOR_TO_PROTO[value] = PROTO.replace(
                _assignment=major.astype(int),
                _timestamp=jnp.array([value]),
                **{
                    k: v[None]
                    for k, v in self_at_this_time.TimeVaryingItems()
                    if (k != "_assignment" and k != "_timestamp")
                },
                is_single_timestamp=False,
            )

        result = MAJOR_TO_PROTO[UNIQUE_TIMESTAMPS[0]]
        # fixme is this correct behaviour: throws away the last one because the last one should have no minors (?)
        for other_self in [MAJOR_TO_PROTO[val] for val in UNIQUE_TIMESTAMPS[1:]]:
            result = result.concat(other_self)
        return result

    def process_raw_minors(self):
        UNIQUE_TIMESTAMPS, UNIQUE_TIMESTAMP_INDEXES, UNIQUE_TIMESTAMP_COUNTS = (
            jnp.unique(self._timestamp, return_counts=True, return_index=True)
        )
        if jnp.all(UNIQUE_TIMESTAMP_COUNTS == 1):
            print(
                "Not doing the sub_timestamp computation because there is no need: all timestamps are unique!"
            )
        else:
            for i, (index, count) in enumerate(
                zip(UNIQUE_TIMESTAMP_INDEXES, UNIQUE_TIMESTAMP_COUNTS)
            ):
                if count == 1:
                    continue

                if i + 1 < len(UNIQUE_TIMESTAMP_INDEXES):
                    slice = self._timestamp.at[
                        UNIQUE_TIMESTAMP_INDEXES[i] : UNIQUE_TIMESTAMP_INDEXES[i + 1]
                    ]
                else:
                    slice = self._timestamp.at[UNIQUE_TIMESTAMP_INDEXES[i] :]

                sliced = slice.get()

                if sliced.shape[0] == 1:
                    continue

                SHAPE_PAD = sliced.shape[0] + 1
                SUB_TIMESTAMP = self.dt / SHAPE_PAD
                incremental = SUB_TIMESTAMP * jnp.arange(SHAPE_PAD)[1:]
                self = self.replace(_timestamp=slice.set(sliced + incremental))
        return self

    def dump_to_parquet(
        self, target_dir, dump_leftover=False, batch_size: int = 100, verbose=True
    ) -> Self:
        """
        Dumps current self to parquet
        Args:
            target_dir:
            batch_size:
            dump_leftover: whether to dump the final leftover data that doesn't fit in a batch_size


        Returns:
            leftover data that didn't fit in a batch_size
        """
        if verbose:
            print(
                "DUMPING TO PARQUET. Note: this function only works ONCE! Call it at the end of data generation."
            )
            print(
                "To make it work for multiple dumping episodes, you need to implement a different sub_timestamp logic:"
            )
            print(
                "right now, it only computes the sub_timestamps by counting all consecutive identical timestamps and"
            )
            print(
                "cumulatively adding 1/NUM_IDENTICAL_TIMESTAMPS to simulate the time taken to navigate the room by the agent"
            )

        self = self.remove_nan_sentinel()
        self = self.process_raw_minors()

        num_full_batches = self._timestamp.shape[0] // batch_size
        leftover_data = self._timestamp.shape[0] - num_full_batches * batch_size

        # todo write some code to find the next id to write in os.path.join(target_dir, f"scan_{i}.parquet"))
        existing = [
            f
            for f in os.listdir(target_dir)
            if f.startswith("scan_") and f.endswith(".parquet")
        ]
        existing_ids = [
            int(re.search(r"scan_(\d+)\.parquet", f).group(1))
            for f in existing
            if re.search(r"scan_(\d+)\.parquet", f)
        ]
        CUR_SCAN_ID = max(existing_ids, default=-1) + 1  # start after the last one

        CUR_IDX = 0
        WROTE_DATA = False
        for batch_id in range(num_full_batches):
            kwargs = {
                k.lstrip("_"): np.array(v)[CUR_IDX : CUR_IDX + batch_size]
                for k, v in self.TimeVaryingItems()
            }
            CUR_IDX += batch_size
            df = pl.DataFrame(kwargs)
            df.write_parquet(os.path.join(target_dir, f"scan_{CUR_SCAN_ID}.parquet"))
            CUR_SCAN_ID += 1
            WROTE_DATA = True

        if leftover_data >= 0 and dump_leftover:
            leftover_data_kwargs = {
                k.lstrip("_"): np.array(v)[CUR_IDX:] for k, v in self.TimeVaryingItems()
            }
            df = pl.DataFrame(leftover_data_kwargs)
            df.write_parquet(os.path.join(target_dir, f"scan_{CUR_SCAN_ID}.parquet"))
            WROTE_DATA = True

        if WROTE_DATA:
            with open(os.path.join(target_dir, "pickupable_names.json"), "w") as f:
                json.dump(self.pickupable_names, f, indent=4)
            with open(os.path.join(target_dir, "receptacle_names.json"), "w") as f:
                json.dump(self.receptacle_names, f, indent=4)
            with open(
                os.path.join(target_dir, "pickupable_to_receptacle.json"), "w"
            ) as f:
                json.dump(self.pickupable_to_receptacle, f, indent=4)
            with open(os.path.join(target_dir, "receptacles_oobb.json"), "w") as f:
                json.dump(self.receptacles_oobb, f, indent=4)
            with open(os.path.join(target_dir, "config.yaml"), "w") as f:
                yaml.dump({"dt": float(self.dt)}, f)
            if self.intrinsics is not None:
                with open(os.path.join(target_dir, "camera_intrinsics.json"), "w") as f:
                    json.dump(self.intrinsics, f, indent=4)

        if not dump_leftover:
            if leftover_data > 0:
                return self.replace(
                    **{k: v[CUR_IDX:] for k, v in self.TimeVaryingItems()}
                )
            else:
                return self.replace(
                    **{
                        k: jnp.zeros_like(v[CUR_IDX - 1 :][0][None]) * jnp.nan
                        for k, v in self.TimeVaryingItems()
                    }
                )
        return None


def load_sssd(path):
    # Load a config file
    split, house_id, jax_key = path_2_parts(path)

    ITEMS = {key: None for key in GeneratedSemiStaticData.TimeVaryingKeys()}

    def extract_index(name: str) -> int:
        base = name.replace(".parquet", "")
        parts = base.split("_")
        if len(parts) == 1:
            return 0
        return int(parts[-1])

    file_names = [f for f in os.listdir(path) if f.endswith(".parquet")]
    file_names = sorted(file_names, key=extract_index)
    file_names = [os.path.join(path, file) for file in file_names]

    with open(os.path.join(path, "pickupable_names.json"), "r") as f:
        pickupables = json.load(f)
    with open(os.path.join(path, "receptacle_names.json"), "r") as f:
        receptacles = json.load(f)
    with open(os.path.join(path, "pickupable_to_receptacle.json"), "r") as f:
        pickupable_to_receptacle = json.load(f)
    with open(os.path.join(path, "receptacles_oobb.json"), "r") as f:
        receptacles_oobb = json.load(f)
    intrinsics = None
    intrinsics_path = os.path.join(path, "camera_intrinsics.json")
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path, "r") as f:
            intrinsics = json.load(f)
    
    try:
        with open(os.path.join(path, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)
            dt = config["dt"]
    except:
        raise Exception(
            "No config.yaml found in the SSSD path, this should ALWAYS exist!"
        )

    return GeneratedSemiStaticData(
        split=split,
        house_id=house_id,
        jax_key=jax.random.PRNGKey(jax_key),
        dt=dt,
        parquet_paths=file_names,
        **ITEMS,
        pickupable_names=pickupables,
        receptacle_names=receptacles,
        pickupable_to_receptacle=pickupable_to_receptacle,
        receptacles_oobb=receptacles_oobb,
        intrinsics=intrinsics,
        is_single_timestamp=False,
        is_single_pickupable=False,
        # batch_size=cfg['jax_scan_size']
    ).resolve_parquet()
