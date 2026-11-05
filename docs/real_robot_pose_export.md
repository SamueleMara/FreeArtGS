# Robot EE / TCP trajectory export

`samuele/export_real_robot_poses.py` reads the original Parquet copied to each
converted real episode under `metadata/raw/`. It writes numeric NumPy archives
under `metadata/robot_poses/`, without changing RGB, depth, masks, or COLMAP.
This is CPU-only; no segmentation or training runs.

Current files contain one measured Cartesian pose in `observation.state`.
`info.json` names its fields `pos_x`, `pos_y`, `pos_z`, `rotvec_x`, `rotvec_y`,
`rotvec_z`. The remaining state values are hand fields, not a second Cartesian
pose. `observation.target_pose` is a control target (zeros in the inspected
examples), and `action.absolute` is a command; neither is exported as measured
TCP ground truth. The user confirmed that the measured pose is EE and supplied
the tool calibrations in
[`real_robot_pose_calibration.json`](../configs/real_robot_pose_calibration.json).

## Export EE and calibrated TCP trajectories

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/export_real_robot_poses.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --parquet-python /home/samuelemara/miniconda3/envs/yolo26/bin/python \
  --calibration-map /home/samuelemara/FreeArtGS/configs/real_robot_pose_calibration.json \
  --overwrite
```

The supplied map applies these EE-local transforms (identity rotation):

| Recordings | Tool | Local TCP +Z |
|---|---|---|
| All existing real recordings except the following three scenes | Pointed tip | 0.1876174 m |
| `white_drawer_20261021_161154_episode_000000`, `000001`, `000002` | Bare force sensor | 0.03487365 m |

The exception entries match the three complete scene names exactly. The map's
default also applies to future scenes; update it for any future tool change.
These values use the supplied assumption that bare contact is at the TCP
origin, relative to the bare contact point. The implied table height in
`base_link`, -0.019542 m, is stored in the configuration as reference only;
it is not part of `T_ee_tcp` and does not shift exported poses. Calibration
was supplied by the user, not independently re-estimated from these recordings.

`--parquet-python` is an installed Python with Polars. It only decodes the
Parquet; the parent process performs NumPy/SciPy geometry. No model runs in
either environment. Add `--dry-run` to validate all records/mappings without
writing. Use `--scene <scene-directory>` instead of `--root` for one episode.
Add `--overwrite` to replace an earlier trajectory export. `--output-dir` is
available with `--scene` for a separate test export.

Files in each scene:

- `poses_full.npz`: all original Parquet observations, sorted by source frame.
- `poses_rgbd.npz`: observations selected by `metadata/real_conversion.json`,
  in the exact output-frame order used by every RGB-D camera.
- `manifest.json`: source hashes, units, coordinate frame, calibration, and
  which EE/TCP arrays are available.

Both archives contain numeric arrays (load with `allow_pickle=False`):

| Array | Shape | Meaning |
|---|---|---|
| `recorded_position_m` | N × 3 | Original Cartesian position in recording base coordinates |
| `recorded_rotvec_rad` | N × 3 | Original orientation as a rotation vector in radians |
| `recorded_transform` | N × 4 × 4 | Local recorded frame → recording base |
| `timestamp_s` | N | Recorded timestamp column, not a synthesized FPS clock |
| `frame_index` | N | Output RGB-D frame ID, or source ID in the full archive |
| `source_frame_index` | N | Original capture frame ID used for joining to Parquet |
| `episode_index` | N | Episode ID |
| `ee_position_m` | N × 3 | Measured EE position in base coordinates |
| `tcp_position_m` | N × 3 | Derived TCP position in base coordinates |
| `ee_transform` | N × 4 × 4 | EE-local → base transform |
| `tcp_transform` | N × 4 × 4 | TCP-local → base transform |

This preserves the full recording rate and also provides the depth-paired
sequence. For example, 458 source observations become 229 RGB-D-aligned rows;
aligned output frame 1 corresponds to source frame 2. No poses are interpolated
or averaged. Hardware synchronization is not inferred from timestamps.

## Identify EE and derive TCP using calibration

After confirming the acquisition convention, supply `--recorded-frame ee` or
`--recorded-frame tcp`. The corresponding `ee_position_m` or `tcp_position_m`
and `ee_transform` or `tcp_transform` are then exported. Without that
confirmation, the default `unknown` exports only the neutral recorded arrays.

To derive the other frame, supply a calibrated rigid transform:

```text
T_base_tcp = T_base_ee @ T_ee_tcp
T_base_ee  = T_base_tcp @ inverse(T_ee_tcp)
```

Use `--ee-to-tcp calibration.json` with a 4×4 JSON matrix, or an object containing
`"T_ee_tcp": <4x4 matrix>`. Its translation must be in meters. For a pure
translation, `--tcp-offset-m X Y Z` specifies the TCP position in EE-local
coordinates and assumes equal EE/TCP orientation. Translation is rotated by the
measured EE orientation; it is not added in world coordinates.

The tools differ across these recordings. To export everything with appropriate
per-recording calibration, fill a copy of
[`robot_pose_calibration.example.json`](../configs/robot_pose_calibration.example.json)
and pass `--calibration-map <filled-file.json>`. Entries under `datasets` match
either the recording prefix (such as `final_black_laptop`) or a full output
scene name. Full scene entries override recording entries; recording entries
override `defaults`. Each entry can define `recorded_frame` and `T_ee_tcp`.
Null calibration means unavailable; it never implies an identity transform.

EE/TCP arrays are omitted when unavailable. The manifest explicitly records
their availability. Approximate tool-axis offsets used for segmentation are
never treated as calibrated TCP transforms.

```python
import numpy as np

path = "/path/to/scene/metadata/robot_poses/poses_rgbd.npz"
with np.load(path, allow_pickle=False) as poses:
    measured_xyz = poses["recorded_position_m"]
    source_frames = poses["source_frame_index"]
    if "ee_position_m" in poses:
        ee_xyz = poses["ee_position_m"]
    if "tcp_position_m" in poses:
        tcp_xyz = poses["tcp_position_m"]
```
