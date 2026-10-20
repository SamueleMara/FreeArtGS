# Real recording to the standard RGB-D/COLMAP layout

For the complete manual command sequence (native recording → RGB/depth → initial/final COLMAP → masks), see [the repository README](../README.md#local-real-rgb-d-recordings-native-format-to-masks).

Run from any directory with the existing environment. The example uses a new `_v2` folder because the original output already exists; full conversion refuses existing output folders:

```bash
python3 -B \
  /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/assets/dataset_fridge_plastic_depth_20261016_163314" \
  --output "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314_v2"
```

This is a CPU conversion using NumPy, OpenCV and SciPy. It does not run segmentation or load any ML models. All generated data stays under `--output`; the source is read only. The output folder must not already exist. A failed conversion retains `conversion_in_progress.json` to identify incomplete data; use a new output path for a retry.

The recording contains three cameras, 655 video frames at 50 FPS, and depth at indices `0, 2, 4, ..., 654`. The converter selects those 328 common source indices, producing 984 synchronized camera observations. Output filenames are consecutive `frame_000000` through `frame_000327` in every modality and COLMAP model. `metadata/real_conversion.json` preserves `source_frames` and `output_frames`; `metadata/frame_timestamps.csv` records output index, original source index, and nominal timestamp. RGB and depth are matched by source index before assigning an output index. Selection is by recorded frame index; capture-time synchronization between physical sensors is not independently established.

Output:

```text
fridge_plastic_20261016_163314/
  rgb/cam_000/frame_000000.png           # similarly cam_001, cam_002; consecutive output indices
  depth_npy/cam_000/frame_000000.npy     # float32 camera-Z, meters
  depth_png/cam_000/frame_000000.png     # uint16 camera-Z, millimeters
  colmap/
    images/cam_000/frame_000000.png      # copies of rectified RGB
    sparse/0/                         # masked, after mask generation
    sparse_unmasked/0/                # pre-mask model
      cameras.txt, cameras.bin
      images.txt, images.bin
      points3D.txt, points3D.bin
  colmap_final/
    images/cam_000/frame_000327.png      # last selected frame from every camera
    sparse/0/                         # masked, after mask generation
    sparse_unmasked/0/                # pre-mask model
      cameras.txt, cameras.bin
      images.txt, images.bin
      points3D.txt, points3D.bin
    export_summary.json
  pointcloud/
    points3D.ply                       # merged first-frame RGB-D cloud
    points3D_final.ply                 # merged last-frame RGB-D cloud
    views/start/cam_000.ply            # per-camera start clouds
    views/end/cam_000.ply              # per-camera end clouds
  metadata/
    cameras.json
    frame_timestamps.csv
    real_conversion.json
    colmap_export.json
    colmap_export_final.json
    raw/                               # original metadata and episode parquet
  README.txt
```

No masks, part labels or joint ground truth are created. Later masks must be generated **from the rectified output RGB**, with matching paths `mask/cam_NNN/frame_NNNNNN.png`. The seed cloud includes the object, robot and background; downstream object training should apply the masks when building object-specific initialization.

RGB is undistorted and resampled to a shared centered PINHOLE camera at 640×480 by default. Focal lengths are the mean native color focal lengths scaled to the requested resolution; all cameras use that same K. Camera optical axes are unchanged. Each raw 640×576 depth image is unprojected using its own calibrated intrinsics and all distortion coefficients, transformed with `T_color_depth`, then projected into this rectified color camera. The closest positive color-camera Z wins when samples hit the same pixel. Missing samples and pixels outside the RGB sensor coverage remain zero; holes are not filled. The converter rejects inverse-distortion rays that fail a reprojection check.

COLMAP poses are world-to-camera inverses of `T_base_color_optical`, with world coordinates in `base_link` and translation in meters. Multiview RGB-D clouds are reconstructed separately at the first and last selected synchronized frames. Per-camera points are transformed into the calibrated base frame, then voxel-fused (5 mm by default), averaging world positions and RGB colors. The initial `colmap/` contains only frame 0 from every camera and its merged first-frame seeds. Both masked and unmasked sparse models contain one image per view, with no other times listed. Their image directories contain only those endpoint images; the full motion sequence remains under `rgb/`, depth, and mask folders. `colmap_final/` contains only the final frame from each camera and its independently merged end-state cloud, matching the endpoint meaning of the synthetic reference. Full conversion uses output frame 327 (source frame 654) as the final endpoint; `--max-frames` instead uses the last frame in that limited selection.

COLMAP tracks retain one source depth observation per camera per voxel; their stored error is the measured reprojection RMS after fusion. These are depth/proximity associations, not feature matches or triangulated landmarks. Per-view clouds are saved under `pointcloud/views/{start,end}/`, and the merged PLYs are `points3D.ply` and `points3D_final.ply`. They are unlabelled; no `parts` filename or fabricated part IDs are added. Temporal states are never fused together. No ICP, pose refinement or bundle adjustment is performed.

The calibration's `CANDIDATE_NOT_VALIDATED` status and original warning are preserved. Conversion does not refine or validate the cross-camera poses. Robot states in the copied parquet are not presented as object joint GT. CSV timestamps are nominal source-frame-index/FPS values, not hardware timestamps.

Useful options:

```text
--dry-run             inspect calibration/videos/depth frame indices, write nothing
--max-frames 2        small conversion test in a separate, new output folder
--size 640 480        output dimensions
--episode 0           recording episode; this recording contains only episode zero
--point-stride 8      sampling stride for each endpoint camera cloud
--voxel-size 0.005    world-space voxel size in meters for endpoint fusion
--final-colmap-only   add the missing final model to an existing converted dataset
```

Validation performed: seven regression tests cover sparse-frame synchronization, rational distortion, depth registration, occlusion/z-buffering, COLMAP binary tracks, distinct initial/final geometry, voxel fusion/color averaging, safe final-only export and unchanged RGB/depth/masks/sources. A real-data final-only smoke test used frame 654 from all three cameras: 9,335 source points became 9,231 fused points, including 88 points with multiview tracks. The upstream COLMAP reader successfully loaded the final binary model with three endpoint images. Full RGB-D conversion has not been rerun after this change.

To add `colmap_final` to the already converted dataset without decoding the recording again:

```bash
python3 -B \
  /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
  --output "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314" \
  --final-colmap-only
```

This mode reads the last converted RGB/depth frame and existing camera metadata, writes the final model/per-view and merged end clouds, and updates conversion metadata. It preserves the existing initial COLMAP model, RGB, depth and masks. It refuses to replace an existing `colmap_final/`.

Run the focused tests:

```bash
python3 -B -m unittest discover \
  -s /home/samuelemara/FreeArtGS/tests -p test_real_recording_conversion.py -v
```


## Generate foreground masks with a configurable prompt

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/create_standard_masks.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314_v2" \
  --prompt "red fridge"
```

`--prompt` is required and passed directly to GroundingDINO. Replace it with the object description for another dataset, for example `--prompt "refrigerator"`. The mask runner reads the converted `rgb/cam_NNN/frame_NNNNNN.png` images and writes matching `mask/cam_NNN/frame_NNNNNN.png` binary uint8 masks (0 background, 255 foreground). It keeps the original image dimensions and calibration. Multiple detections for the supplied description are combined into one foreground mask per image; the best SAM2 hypothesis is used for each detected box.

CUDA is used by default, with local GroundingDINO and SAM2.1 base-plus weights and an existing BERT cache. No models are downloaded. Only segmentation uses the `grounded-sam2` environment; CPU geometry/export steps use host Python. Runtime caches go under the converted dataset's `.runtime/`. Source RGB/depth are not modified. After a complete successful mask pass, the shared Sapien filter publishes foreground models to `colmap/sparse/0` and `colmap_final/sparse/0`; their pre-mask models stay in `sparse_unmasked/0`.

Matching completed masks are reused automatically. `metadata/mask_generation.json` records the prompt, thresholds, source RGB checksums, detections, foreground pixel counts and empty masks. A changed prompt/settings requires `--overwrite`; existing unrecorded masks are not silently overwritten. Empty detections still get an empty mask file so indexing remains complete, but the command returns nonzero and identifies them in the report. Inspect automatic masks before training, especially if the prompt describes only a door instead of the whole object.

`--dry-run` lists the planned frame count without model loading or writes. `--max-frames 1` tests one image per camera. Thresholds can be adjusted with `--box-thr` and `--text-thr` (both default to 0.25). A full run after a limited test reuses the tested frames.


Six focused mask-runner regression tests cover configurable prompts, standard paths and binary dimensions, unchanged RGB files, matching-run reuse, changed prompts/RGB rejection, empty-mask reporting, dry runs and per-camera limits:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B -m unittest discover \
  -s /home/samuelemara/FreeArtGS/tests -p test_standard_masks.py -v
```

## Repair outputs created with sparse capture indices

This renames existing RGB, depth PNG/NPY, masks, and both COLMAP image trees
to consecutive zero-based names. It also updates COLMAP text/binary image
names, mask manifest keys, endpoint metadata, and timestamp rows. Data files
are moved without re-encoding; SHA256 verification checks every file. Original
reference files and the resumable repair journal remain under
`metadata/frame_renumbering_backup/`. Source recording metadata stays intact.

```bash
source /home/samuelemara/miniconda3/etc/profile.d/conda.sh
# Use host Python for CPU conversion/export.
python3 -B /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
  --output "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314" \
  --renumber-frames
```

Add `--dry-run` to inspect without changes. If interrupted, rerun the same
command to resume. Do not train or generate masks while the repair is active.

## Masked and unmasked COLMAP models (initial and final)

Each endpoint now has two sparse models:

```text
<scene>/
  colmap/
    sparse/0/             # masked, used by training
    sparse_unmasked/0/    # full pre-mask model
    images/
  colmap_final/
    sparse/0/             # masked final state
    sparse_unmasked/0/    # full pre-mask final state
    images/
```

Both sparse folders contain `cameras`, `images`, and `points3D` in text and
binary formats by default. The shared `images/` directories retain the original
RGB images. A 3D point survives if at least one of its recorded observations is
inside its corresponding `mask/<camera>/<frame>.png`. Background observations
are detached; points without foreground support are removed. Camera poses,
point coordinates/colors, IDs, and observation indices are preserved. Missing,
wrong-size, or nonbinary masks are errors; an empty filtered model is rejected.
The image/point tracks remain consistent in both text and binary models.

Fresh SAPIEN exports apply this automatically using the rendered masks. Fresh
real conversion first writes **only `sparse_unmasked/0`**; complete, successful
mask generation then publishes `sparse/0`. This prevents training from silently
using unmasked geometry while masks are still missing. The host-side real-data
mask runner uses this repository's `scripts/mask_colmap_models.py` (default
checkout `/home/samuelemara/Joint3DGS/Sapien`; set `SAPIEN_REPO` if moved).

For **existing outputs**, run the following commands on the host. These perform
CPU-only filtering of stored models, reconstructing missing synthetic endpoints
from saved RGB-D when requested: no rendering, model inference, or Docker/GPU
is needed. They preserve RGB/depth/masks and existing PLY files;
the masked selection is saved in the COLMAP sparse models. Synthetic pre-mask
models already exclude actors without articulated-object part labels, so some
scenes may retain all existing points.

```bash
source /home/samuelemara/miniconda3/etc/profile.d/conda.sh
# Use host Python for CPU conversion/export.

# Existing real recording: both initial and final endpoints.
python3 -B /home/samuelemara/Joint3DGS/Sapien/scripts/mask_colmap_models.py \
  --scene "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314"

# All 60 PARIS synthetic scenes, using their existing masks and saved RGB-D.
SAPIEN_OUTPUT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output"
batch_status=0
for dataset in "$SAPIEN_OUTPUT"/preparation_8cams/partnet_subset_fixed_render* "$SAPIEN_OUTPUT"/preparation_64cams "$SAPIEN_OUTPUT"/preparation_100cams; do
  [ -d "$dataset" ] || continue
  python3 -B /home/samuelemara/Joint3DGS/Sapien/scripts/mask_colmap_models.py \
    --root "$dataset" --build-missing-endpoints || batch_status=1
done
[ "$batch_status" -eq 0 ]
```

`--root` discovers synthetic scenes by `metadata/job_config_resolved.yaml`, so
real recordings are handled separately. Add `--dry-run` to either command for
read-only validation and point counts. Matching masks must already exist. Endpoint models must exist unless
`--build-missing-endpoints` is supplied for saved synthetic RGB-D. Failed scenes are reported, other scenes
are still processed, and the command exits nonzero if any fail.

The first existing-data run copies the original sparse models into
`sparse_unmasked/0`. Later runs always start from those saved originals, so
edited masks can restore previously excluded points. Initial and final masks
are validated before publishing either endpoint. Summaries record both point
counts, the filtering rule, and hashes of all masks used. Do not train or modify
masks while publishing new sparse models. Existing `pointcloud/*.ply` remain
pre-mask exports; use `sparse/0/points3D.{txt,bin}` for the filtered points.

### Missing synthetic endpoint models: use saved RGB-D on CPU

The synthetic datasets already contain rendered masks; no SAM or mask inference
is needed. Their saved depth and existing sparse points may already contain
only foreground, so the masked and unmasked copies can be identical.

The current SSD has 20 PARIS scenes missing final models. Pass
`--build-missing-endpoints` to reconstruct missing endpoints from saved metric
`depth_npy`, RGB, and camera calibration. This does not load SAPIEN, rerender,
change masks, or invent background or part labels. It requires the true first
and last saved frames, and publishes both masked and unmasked COLMAP models.
Existing PLY files are preserved. Endpoint reconstruction samples depth with
stride 8 and a maximum of 1000 points per camera, matching exporter defaults.

The synthetic command above covers all 60 PARIS scenes: four 8-camera profiles
and the flat 64/100-camera roots. It excludes the unrelated `franka_joint1`
sample. Both filtering and missing-endpoint reconstruction run on the host CPU.
