# FreeArtGS: Articulated Gaussian Splatting Under Free-moving Scenario

<div align="center">
  <p>
    <a href="https://daihangpku.github.io/">Hang Dai<sup>*</sup></a>,
    <a href="https://hwfan.io/about-me/">Hongwei Fan<sup>*</sup></a>,
    <a href="https://github.com/Zarad0X">Han Zhang<sup>*</sup></a>,
    <a href="https://github.com/Wu-dolores">Duojin Wu</a>,
    <a href="https://jiyao06.github.io/">Jiyao Zhang</a>,
    <a href="https://zsdonghao.github.io/">Hao Dong</a>
  </p>
</div>
<div align="center">
  (* indicates equal contribution)
</div>
<div align="center">
  <strong>CVPR 2026</strong>
</div>
<div align="center">
<a href="https://arxiv.org/abs/2603.22102">
  <img src="https://img.shields.io/badge/arXiv-2603.22102-b31b1b" alt="arXiv">
</a>
<a href="https://freeartgs.github.io/">
  <img src="https://img.shields.io/badge/Project_Page-FreeArtGS-green" alt="Project Page">
</a>
</div>

<!-- <p align="center">
    <img src="assets/method.png" alt="Teaser" style="center" />
</p> -->

This repository contains the official implementation of [FreeArtGS: Articulated Gaussian Splatting Under Free-moving Scenario](https://freeartgs.github.io/).

<p align="center">
  <img src="docs/pipeline.png" alt="Overview of FreeArtGS" width="90%">
</p>


## Installation

Please follow [docs/install.md](docs/install.md) to set up the environment and required third-party dependencies.

This fork stores its modified Nerfstudio implementation in the `nerfstudio-local`
branch of the same repository. Clone with submodules to obtain the matching code:

```bash
git clone --recurse-submodules git@github.com:SamueleMara/FreeArtGS.git
```

For an existing checkout, run `git submodule sync --recursive` followed by
`git submodule update --init --recursive` after pulling this fork.

## Data Preparation

### Local real RGB-D recordings: native format to masks

The local tools convert the native `videos/`, `depth/`, `meta/`, and `data/`
recording into the standard RGB-D/COLMAP layout, then generate foreground masks.
Run these commands yourself, in order. The complete format and options are
documented in [Real recording conversion](docs/real_recording_conversion.md).

Set the recording, a **new output folder**, and the segmentation prompt:

```bash
REAL_INPUT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/assets/dataset_fridge_plastic_depth_20261016_163314"
REAL_OUTPUT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314_v2"
MASK_PROMPT="red fridge"
```

The original `fridge_plastic_20261016_163314` output already exists. The example
uses `_v2` to keep it intact; choose another unused folder if necessary.

1. Preview the conversion without writing files:

   ```bash
   python3 -B \
     /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
     --input "$REAL_INPUT" --output "$REAL_OUTPUT" --dry-run
   ```

2. Convert RGB/depth and build both COLMAP models (CPU):

   ```bash
   python3 -B \
     /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
     --input "$REAL_INPUT" --output "$REAL_OUTPUT" --point-fraction 1
   ```

   This writes rectified `rgb/`, registered `depth_npy/` in meters,
   `depth_png/` in millimeters, camera metadata, `colmap/`, and `colmap_final/`.
   The initial and final unmasked seed clouds merge the cameras separately at
   the first and last selected frames, using 5 mm voxels. Per-camera clouds and merged
   `points3D.ply` / `points3D_final.ply` are saved under `pointcloud/`.
   The output folder must not already exist; no source files are changed.

3. Generate masks from the converted RGB (CUDA):

   ```bash
   /home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
     /home/samuelemara/FreeArtGS/samuele/create_standard_masks.py \
     --input "$REAL_OUTPUT" --prompt "$MASK_PROMPT"
   ```

   Change `MASK_PROMPT` to describe the whole object. The required `--prompt`
   argument controls detection; it is not hardcoded. Masks are written as
   `mask/cam_NNN/frame_NNNNNN.png`, binary 0/255, at the converted RGB resolution.
   Matching completed masks are reused. Empty masks are identified in
   `metadata/mask_generation.json` and produce a nonzero exit status.

To regenerate masks in the **existing** converted dataset, including those
already generated with the same prompt:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/create_standard_masks.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314" \
  --prompt "red fridge" --overwrite
```

`--overwrite` replaces masks, so omit it to reuse matching results. Changing a
previously used prompt or thresholds requires `--overwrite`. Masks must use the
rectified output RGB, not the native MP4 frames. Calibration retains its original
`CANDIDATE_NOT_VALIDATED` status; no joint GT is fabricated.

The real converter saves pre-mask COLMAP models in `sparse_unmasked/0` for
both endpoints. A complete successful mask-generation pass publishes masked
models to `sparse/0`. Real masked models are reconstructed directly from saved
registered depth, RGB, and endpoint poses, without voxel averaging. The shared
COLMAP utilities require the Sapien checkout at `~/Joint3DGS/Sapien`, or the path
set in `SAPIEN_REPO`.
See [masked/unmasked COLMAP commands](docs/real_recording_conversion.md#masked-and-unmasked-colmap-models-initial-and-final)
for CPU-only updates to existing real and synthetic datasets.

### Robot dog masks: distinguish the dog from the manipulator

Text-only prompts `robot dog`, `robotic arm`, and `end effector` can all select
both robots. Subtracting those masks can erase the entire dog. For this specific
640×480 fixed-camera recording, use the explicit per-camera SAM2 boxes and
positive/negative points in `configs/masks/robot_dog_20261022_185923.json`:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/create_standard_masks.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/dog_20261022_185923_episode_000000" \
  --prompt "robot dog" \
  --prompt-points /home/samuelemara/FreeArtGS/configs/masks/robot_dog_20261022_185923.json \
  --device cuda --overwrite
```

This runs SAM2 directly: the text detector and text-mask subtraction are
bypassed. The prompt is the object label; the JSON identifies foreground and
background spatially. Masks must satisfy the supplied point labels, and are
clipped to the specified object box. The guidance must match every camera's RGB
resolution. Its hash and applied points are recorded in the mask manifest.
Changing the guidance requires `--overwrite`. Use different guidance for another
recording or moving cameras. Inspect masks near contact and occlusion boundaries.

If interrupted, rerun the same command with **`--resume` instead of `--overwrite`**.
It requires the same prompt/guidance settings, reuses recorded completed masks,
and regenerates missing or unrecorded leftovers from older runs. It does not
restart the completed cameras. A failed single-mask prediction is retried using
alternate SAM candidates, then up to two mask-logit refinements; each accepted
output must still satisfy every explicit point label after box clipping.
Unresolved violations stop the run with the camera/frame and offending points,
instead of silently publishing an invalid mask. The manifest records refinement
retry counts. Mask generation publishes both masked COLMAP endpoints only after
the full sequence completes successfully.

For other scenes, `--exclude-prompt "object to remove"` remains available, with
separate exclusion masks under `mask_excluded/`. It now rejects a subtraction
that erases a nonempty target, rather than publishing that empty mask. The
per-frame exclusion detections and removed-pixel counts are recorded in the
manifest. Explicit point guidance cannot be combined with exclusion prompts or
Leica guidance. A full successful mask pass rebuilds both masked COLMAP
endpoints and matching PLYs with the saved point fraction.

### Choose real RGB-D point-cloud and COLMAP downsampling

`--point-fraction` selects a fraction of valid depth pixels **per camera**, for
both initial and final endpoints. `1` keeps 100%, `0.25` keeps 25%, and `0.125`
keeps one eighth. Sampling is deterministic across valid pixels, retaining
`floor(N * fraction)` points (at least one if a view has valid points).
An 8-pixel stride in both image dimensions is approximately **1/64**, not 1/8.

For new conversions, the fraction replaces the legacy `--point-stride 8` grid
sampling. Unmasked seeds still undergo `--voxel-size` fusion (default 5 mm), so
their merged point count can be smaller. The fraction is saved in metadata and
used after mask generation. Masked exports sample only valid pixels inside each
mask, do not voxel-merge, and write matching PLY and COLMAP points.

Rebuild an existing scene using its saved RGB/depth/masks, without running SAM:

```bash
MASK_PYTHON="/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python"
REAL_ROOT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
"$MASK_PYTHON" -B /home/samuelemara/FreeArtGS/samuele/colmap_masking.py \
  --scene "$REAL_ROOT/fridge_plastic_20261016_163314" --point-fraction 0.125

# Both endpoints of all six newer real scenes:
for SCENE in "$REAL_ROOT"/*_20261021_161154_episode_00000[0-5]; do
  "$MASK_PYTHON" -B /home/samuelemara/FreeArtGS/samuele/colmap_masking.py \
    --scene "$SCENE" --point-fraction 0.125 || break
done
```

This replaces masked `colmap/sparse/0`, `colmap_final/sparse/0`, and
`pointcloud/points3D.ply` / `points3D_final.ply`. Matching PLYs also live inside
each masked sparse folder. Original unmasked models are preserved, with separate
`points3D_unmasked.ply` / `points3D_final_unmasked.ply` exports. Existing-scene
rebuilds leave the seed sampling and `pointcloud/views/` unchanged. Omitting the
fraction reuses the saved setting, or 1 for older recordings; pass `1` explicitly
to restore every valid masked depth pixel.

To convert all six native episodes and generate masks with the assigned prompts:

```bash
"$MASK_PYTHON" -B /home/samuelemara/FreeArtGS/samuele/prepare_real_dataset_batch.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/assets/dataset_fridge_plastic_depth_20261021_161154" \
  --outputs-root "$REAL_ROOT" --point-fraction 0.125 --dry-run
```

Remove `--dry-run` to process. Add `--overwrite` only when intentionally replacing
the six existing named scene folders; all selected inputs validate before any
folder is removed. Episodes 0–2 use `white drawer`, 3 uses `black box`, 4 uses
`laptop`, and 5 uses `black monitor`. A mask failure is reported and does not skip
subsequent episodes. The fraction can also be supplied to
`create_standard_masks.py`; changing it reuses matching masks.
This option applies to metric real recordings; synthetic/HoI exporters retain
their existing sampling. Details: [real point-cloud exports](samuele/DENSE_REAL_COLMAP.md).

### Batch preparation of named real recordings

Use `--inputs` for recordings such as `final_black_laptop` or `final_robot_dog`.
The batch discovers every episode listed in each source's `meta/episodes.jsonl`;
there is no six-episode limit. Prompts come from the folder name: remove
`final_`, replace underscores with spaces, and normalize `suitecase` to
`suitcase`. Source task labels are ignored. This first pass uses text-only SAM
masking; the old robot-dog spatial configuration is not applied to new captures.

```bash
ASSETS="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/assets"
OUTPUTS="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/prepare_real_dataset_batch.py \
  --inputs \
    "$ASSETS/final_black_laptop" \
    "$ASSETS/final_matchbox" \
    "$ASSETS/final_robot_dog" \
    "$ASSETS/final_black_box" \
    "$ASSETS/final_light_blue_box" \
    "$ASSETS/final_orange_suitecase" \
    "$ASSETS/final_wooden_box" \
    "$ASSETS/final_red_fridge" \
  --outputs-root "$OUTPUTS" --point-fraction 1 --resume
```

Add `--dry-run` to validate without writing. The current eight recordings contain
110 episodes, all passing RGB-D/calibration preflight. Each output is named
`<source_folder>_episode_000000`, preserving the source name even when correcting
a spelling error in the prompt. Conversion uses all calibrated cameras and
produces initial/final unmasked seeds; successful mask passes publish both masked
COLMAP endpoints and matching PLYs. Only outputs and runtime caches under the
chosen output root are written.

`--resume` skips matching completed conversions and reuses recorded masks;
unrecorded leftovers from interrupted mask runs are regenerated. An incomplete
conversion is reported for explicit inspection/rebuilding, not deleted. Use
`--overwrite` instead only to deliberately rebuild the selected scene folders.
Mask failures do not skip later episodes, and produce a nonzero batch exit code.
Review empty masks in `metadata/mask_generation.json` and inspect visual quality;
nonempty masks can still contain the wrong object. Spatial correction profiles
for these new captures should be prepared separately after that review.
The legacy single `--input` mode retains the earlier six-episode prompt mapping.

### Correct masks in existing real collections

Reviewed profiles for monitor, matchbox, laptop, robot dog, wooden box, black box,
and orange suitcase live
under `configs/masks/`. They select the actual small object rather than the support
box and use object-specific foreground/background points. Matchbox and basket
profiles crop the input around the object before SAM2 inference, then restore
the mask to the original calibrated 640×480 image. A refinement pass keeps the
valid candidate with the better SAM score; that score is not ground-truth IoU.

The current selection includes 84 newer episodes plus the older monitor,
laptop, and black-box recordings (87 scenes). This operates on existing standard outputs:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/correct_real_masks.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --groups monitor matchbox laptop robot_dog wooden_box black_box orange_suitcase \
  --overwrite
```

Add `--dry-run` for scene/profile checks without writing or inference. To continue
an interrupted correction, replace `--overwrite` with `--resume`; completed masks
with matching guidance are reused. Limit `--groups` to any subset to correct only
those objects. A scene failure is reported and other scenes continue. Inspect
`mask_correction_report.json` under the root for completed/failed scenes.

To correct only black boxes, use `--groups black_box`. This selects the twelve
`final_black_box_episode_*` scenes and the older
`black_box_20261021_161154_episode_000003` scene, with separate profiles for
their different object heights. The profiles reject cardboard supports and
retain the case body and opening lid using cropped, per-camera spatial prompts.
The older recording's `cam_001` uses the full image so SAM2 can distinguish
the robot from the case at contact. To run just this correction:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/correct_real_masks.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --groups black_box --overwrite
```

Black-box validation covers initial, middle, and final frames from all three
cameras across 13 scenes (117 samples). Point-constraint checks do not establish
ground-truth segmentation accuracy; small tool-contact boundary errors remain
possible, and intermediate frames are not exhaustively checked.

To restore the moving telescopic handle and rails on all twelve orange-suitcase
episodes, use the dedicated profile:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/samuele/correct_real_masks.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --groups orange_suitcase --overwrite
```

The recording directory is spelled `final_orange_suitecase`; the command group
and object label use `orange_suitcase` / `orange suitcase`. Bounds include the
fully extended handle, and foreground anchors at the rail attachments keep
the silver rails and black grip with the suitcase. Cardboard-support prompts
exclude the stand. Robot-pose prompts are disabled because the projections
are inconsistent across these episodes. Small contact-boundary errors are
still possible; inspect the grip near the robot before training. Validation
checked five times per camera across all twelve episodes (180 sampled frames);
it does not cover every frame or measure ground-truth segmentation accuracy.

SAM2 is loaded once on CUDA in the `grounded-sam2` process. Where verified,
recorded tool poses provide moving background prompts through the saved camera
calibration. Tool-axis extent is specific to the attachment; the monitor uses
a shorter extent. Robot-dog and basket profiles use image-space prompts because
the pose projections were unreliable there. These are segmentation prompts,
not object pose or joint GT. Validated projected prompts are saved in each
scene's `metadata/robot_mask_guidance.json`; source-pose/calibration hashes are
checked before reuse. For uncached poses, an existing Python with `polars` can
be specified solely for reading the raw Parquet, for example:

```bash
# Add this argument only if the projected-pose cache is missing or stale:
--parquet-python /home/samuelemara/miniconda3/envs/yolo26/bin/python
```

No model runs in that reader environment. The correction replaces masks and
rebuilds both masked COLMAP endpoints and their matching PLYs. It preserves
source recordings, converted RGB/depth, and unmasked COLMAP models. Runtime
files and the correction report remain under the target root.

The original five groups passed 558 sampled frame checks across 62 scenes; not
every intermediate frame was tested. Before/after overlays and the validation
report are saved under `mask_correction_previews/`. Inspect occlusion boundaries,
thin handles, and gripper contacts; passing point constraints does not prove a
perfect silhouette.

### Export recorded robot positions and calibrated EE/TCP trajectories

Use `samuele/export_real_robot_poses.py` to extract Parquet poses into NumPy
archives for every converted real episode:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/export_real_robot_poses.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --parquet-python /home/samuelemara/miniconda3/envs/yolo26/bin/python \
  --calibration-map /home/samuelemara/FreeArtGS/configs/real_robot_pose_calibration.json \
  --overwrite
```

Each scene receives `metadata/robot_poses/poses_full.npz` (all Parquet rows),
`poses_rgbd.npz` (exact RGB-D frame alignment), and `manifest.json` with units,
source hashes, and pose availability. This is CPU-only. Add `--dry-run` to
validate without writing and `--overwrite` to replace an existing export.

The measured pose is user-confirmed EE. The supplied calibration map derives
TCP using EE-local +Z = 0.1876174 m (pointed tip), except the three
`white_drawer_20261021_161154_episode_00000{0,1,2}` scenes, which use
0.03487365 m (bare force sensor). Both `ee_position_m` and `tcp_position_m`
are exported in base coordinates. The implied table Z = -0.019542 m is
reference metadata and does not shift either trajectory. `--overwrite` replaces
only the pose exports. See [robot pose export documentation](docs/real_robot_pose_export.md)
for arrays, calibration assumptions, and custom tool mappings.

### Synthetic dataset FPS metadata

Synthetic scenes record playback rate as `fps` in `metadata/cameras.json` and
`metadata/timing.json`, and as `fps_resolved` in
`metadata/job_config_resolved.yaml`. The current datasets use **59.0 FPS**.
All 328 saved videos across the four 8-camera profiles and the Franka example
report that rate. The 64- and 100-camera scenes have no saved videos; their rate
is recorded with `fps_source: provided_video_estimate` rather than claiming a
per-scene measurement. Timing metadata also records frame interval and count.

To validate and backfill all synthetic scenes, across camera counts and motion
profiles, without rerendering:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/add_synthetic_fps.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output" \
  --fps 59 --dry-run
```

Remove `--dry-run` to update metadata. All scenes validate before any writes;
video rates that disagree with `--fps` are errors. Real and HoI datasets are
excluded. RGB, depth, point clouds, COLMAP models, and joint trajectories stay
unchanged. The render-scheduling `fps` field in YAML stays unchanged because
it cannot be combined with a fixed `num_frames`; `fps_resolved` records the
playback rate separately. The local Sapien renderer now writes this metadata
for future exports using its resolved video FPS.

### Official datasets

Download FreeArt-21:

```bash
pip install gdown
gdown 1hl6s3C3919gXKvHC0ZfgrxWqFCKM3XXf -O ./
unzip ./FreeArt-21_v1.zip -d .
```

Download the MonoMobility evaluation data:

```bash
gdown 1K0z1_Gtk9G_51_o61OAhBjo1hxrDCFey -O ./
unzip ./data.zip -d .
```

## Quick Start

For the multi-camera datasets already in this workspace, see
[Joint training with the local datasets](docs/local_datasets.md) to train each
object using all eight calibrated cameras in one shared model.
The local batch stores results, prepared datasets, converted inputs, configs and
runtime caches under the SSD target selected by `--outputs-root`.

FreeArtGS is driven by a single YAML config. The example below runs object `100109` from FreeArt-21 with [configs/freeart-21.yaml](configs/freeart-21.yaml).

### 1. Select the Matplotlib Backend

Set `MPLBACKEND` before running the pipeline. For interactive local runs:

```bash
export MPLBACKEND=TkAgg
```

For headless/server runs:

```bash
export MPLBACKEND=Agg
```

### 2. Run the Full Pipeline

The full pipeline performs preprocessing, two-part tracking, reconstruction, blending, articulation optimization, and evaluation:

```bash
bash pipelines/freeartgs.sh --config configs/freeart-21.yaml --object_name 100109
```

Intermediate and final results will be saved under `datasets/100109/` and `outputs/100109/`.

You can also run individual stages:

```bash
bash pipelines/freeartgs.sh --config configs/freeart-21.yaml --object_name 100109 --preprocess_only
bash pipelines/freeartgs.sh --config configs/freeart-21.yaml --object_name 100109 --tracking_only
bash pipelines/freeartgs.sh --config configs/freeart-21.yaml --object_name 100109 --reconstruction_only
```

### 3. Evaluate an Existing Result

After reconstruction has finished, run:

```bash
bash pipelines/evaluate_freeartgs.sh --object_name 100109
```

### 4. Visualize Results

Visualize the two-part blend result:

```bash
python reconstruction/visualize_twopart_blend.py \
  --input-dir outputs/100109/twopart_blend \
  --motion_part 0
```

Use a hard part mask instead of the default soft blending:

```bash
python reconstruction/visualize_twopart_blend.py \
  --input-dir outputs/100109/twopart_blend \
  --motion_part 0 \
  --hard \
  --threshold 0.5
```

Visualize the final articulated Gaussian result:

```bash
python reconstruction/visualize_articulation.py \
  --input-dir outputs/100109/splatfacto-art \
  --motion_part 0 \
  --soft
```

The articulation viewer can render with GSplat and record frames periodically:

```bash
python reconstruction/visualize_articulation.py \
  --input-dir outputs/100109/splatfacto-art \
  --motion_part 0 \
  --use-gsplat-render \
  --record \
  --record-interval 0.1
```

Common controls: `A/D` step backward/forward, `R` reset, `M` toggle the moving part, `J/K` adjust the mask threshold, and `S` toggle soft/hard visualization. The articulation viewer also supports `P` to save a capture and axis/origin adjustment keys printed in the terminal.

## Open-source TODO List

- [x] Release installation instructions.
- [x] Release FreeArt-21 data.
- [x] Release the main FreeArtGS pipeline config and scripts for FreeArt-21.
- [x] Release evaluation scripts.
- [x] Release visualization tools.
- [ ] Release the pipeline config and scripts for real data.
