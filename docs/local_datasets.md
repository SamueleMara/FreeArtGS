# Joint training with the local calibrated datasets

This is an experimental adaptation of the official monocular RGB-D pipeline.
The review in [official_pipeline_review.md](official_pipeline_review.md) found
incorrect part labels under occlusion. Training completion and a small axis
angle do not establish correct reconstruction. The shared models use all views,
but the independent segmentation frontend still needs correction.

Run from any directory:

```bash
bash /home/samuelemara/FreeArtGS/pipelines/train_local_datasets.sh --all-cameras
```

This discovers all raw object folders and trains one shared articulated model
per object. Use `--objects` to select an exact list and its processing order.
Each object uses all eight cameras and all 60 synchronized timestamps, giving
480 observations per model. `episode_000000` and generated datasets are excluded.
Objects run sequentially; each object's camera views contribute to the same
Gaussian geometry, part weights, joint axis and joint origin. All views at a
timestamp share the same object pose and final joint parameter. Optimization
samples one observation per step, with images cached on the CPU.
Training uses Tensorboard logging and does not open an interactive viewer.

The launcher activates the installed `freeartgs` Conda environment and CUDA 12.8.
It validates all raw inputs before training. Each of the two part models, the
combined model and the final articulated model trains for 30,000 iterations.
The default target is `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/freeart`.
All generated files, library caches and temporary files now stay under that
target. `--datasets` selects read-only raw sources; it no longer selects where
prepared datasets are written. `--outputs-root` overrides the entire target.
The default raw source is now the existing SSD backup at
`/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render`.
The former `FreeArtGS/datasets` source directory disappeared during a run.
All 180 converted oven RGB/depth/mask observations matched this backup exactly.
The runner reports missing source paths clearly and stops if a source disappears
after preflight. Keep raw source folders available throughout training.

To train the ten requested objects, evaluate their joints against ground truth,
and save motion comparison videos in per-object output folders:

```bash
bash /home/samuelemara/FreeArtGS/pipelines/train_local_datasets.sh \
  --datasets "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render" \
  --outputs-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/freeart" \
  --all-cameras --evaluate-joint-axis --visualize-results \
  --objects foldchair_102255 laptop_10211 oven_101917 scissor_11100 \
    washer_103776 storage_45135 blade_103706 fridge_10905 \
    stapler_103111 USB_100109
```

`--visualize-results` uses each completed model's saved checkpoint renders to
create MP4/GIF comparisons with estimated axes. It also evaluates joint GT when
`pointcloud/joint_metadata.json` exists. Real recordings without joint GT show the
estimated axis on both recorded RGB and model renders. `--evaluate-joint-axis`
explicitly requires GT. If comparison creation
fails, the summary preserves the completed training result, records
`visualization_error`, continues with later objects and returns a nonzero exit
status at the end. Storage failures stop the batch immediately. Missing GT states
or a joint type mismatch omit the GT state comparison but still allow axis videos.

GT comes from each raw object's `pointcloud/joint_metadata.json` in `render_world`
coordinates. After training, the evaluator uses the calibrated and exported camera
rotations to bring the predicted axis into that same frame. It reports the unsigned
angle `acos(abs(dot(pred_axis, gt_axis)))` in degrees, treating an axis and its
negative as equivalent. Each result is saved to
`<target>/<object>_multiview/splatfacto-art/joint_axis_error.json` and included in
`<target>/batch_logs/summary.json`; the console also prints the mean over evaluated
objects. GT is used only for evaluation. This metric measures axis direction,
not the location of the axis line.
The report also evaluates joint type and relative states from
`metadata/frame_values.csv`, matched through source timestamps, and reports
revolute axis-line position error and camera position consistency. Relative
states remove only the first-state offset and use the predicted axis sign;
they do not fit a scale to GT. These are local diagnostics, not the official
FreeArt-21 benchmark, and do not assess geometry or held-out render quality.
When the exported PLY and joint origins are available, evaluation also saves
`joint_axis_preview.png`, showing Gaussian centres with the predicted axis in red
and GT in green. For prismatic joints the origin does not affect translation;
the angular score does not constrain its location.

Open the completed storage model and its axes from a desktop terminal:

```bash
freeart_target="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/freeart"
eval "$(/home/samuelemara/miniconda3/envs/freeartgs/bin/python -B /home/samuelemara/FreeArtGS/utils/local_storage.py --outputs-root "$freeart_target")"
bash /home/samuelemara/FreeArtGS/pipelines/view_local_articulation.sh --input-dir "$freeart_target/storage_45135_multiview/splatfacto-art" --motion_part 1 --use-gsplat-render
```

The interactive Open3D view displays Gaussian centres and the two axes.
Press `P` to save a Gaussian splat render; `A`/`D` change the articulation and
`R` resets it. Captures go to `viewer_exports/` inside the model output directory.
The PLY viewer uses a simplified part assignment/animation and does not reproduce
the checkpoint's soft opacity mixture exactly. Actual model renders are saved
in `splatfacto-art/eval_results/` after training. The storage comparison in
`outputs/dataset_validation/storage_gt_render_comparison.png` shows all eight
views at the first and last timestamps alongside their inputs.

Generate a motion video comparing all cameras at every saved timestamp:

```bash
freeart_target="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/freeart"
eval "$(/home/samuelemara/miniconda3/envs/freeartgs/bin/python -B utils/local_storage.py --outputs-root "$freeart_target")"
freeart_sources="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render"
/home/samuelemara/miniconda3/envs/freeartgs/bin/python -B evaluate/visualize_local_comparison.py --scene "$freeart_sources/storage_45135" --dataset "$freeart_target/work/datasets/storage_45135_multiview" --results "$freeart_target/storage_45135_multiview/splatfacto-art"
```

Outputs are `splatfacto-art/comparison/motion_vs_gt.mp4` and `motion_vs_gt.gif`,
with GT RGB on the left and saved checkpoint renders on the right for each camera.
Raw GT foreground masks composite the RGB onto white; foreground pixels are
preserved. Green arrows show the GT axis direction, red arrows the learned
direction. For a prismatic joint, both directions use the same display anchor;
this is not an estimate of pivot position. These are training-view comparisons,
not novel-view evaluation or fresh GPU renders.

### Estimated-axis debug for real recordings without joint GT

The same visualization command works without `pointcloud/joint_metadata.json`.
It uses the exported cameras to map the predicted axis and origin into the source
COLMAP calibration frame, then projects them into every camera. Full recorded RGB
is shown on the left and the saved model render on the right, both with the
estimated axis in red with a white outline. The header shows the estimated motion
relative to the first timestamp, in degrees or meters. Axis lines are overlays
and remain visible through occluders; they are not depth-tested geometry.

For the completed real fridge run, generate the debug without retraining:

```bash
freeart_target="/media/samuelemara/Extreme SSD/Samuele/results_18_10_26/freeart_real"
freeart_object="fridge_plastic_20261016_163314_multiview"
/home/samuelemara/miniconda3/envs/freeartgs/bin/python -B \
  /home/samuelemara/FreeArtGS/evaluate/visualize_local_comparison.py \
  --scene "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314" \
  --dataset "$freeart_target/work/datasets/$freeart_object" \
  --results "$freeart_target/$freeart_object/splatfacto-art" \
  --output-dir "/home/samuelemara/FreeArtGS/outputs/real_fridge_analysis_20261019/axis_debug" \
  --fps 25
```

This writes `motion_axis_debug.mp4`, `motion_axis_debug.gif`, first/middle/last
`comparison_*.png` stills and `comparison_info.json` with the calibrated predicted
axis/origin. No GT axis, origin or error is invented. When joint GT exists, the
existing `motion_vs_gt.*` outputs and optional GT state comparison are retained.
For a prismatic joint the display anchor is arbitrary, not an estimated pivot.
Without `--output-dir`, outputs go to `<results>/comparison/`.
The recorded sequence has 25 selected timestamps per second; `--fps 25` preserves
its nominal timing. Other recordings should use their own selected frame rate.

The final learned parameters are in `splatfacto-art/articulation.json` in the
model export frame. Compare `predicted_axis` and `gt_axis` in
`splatfacto-art/joint_axis_error.json`, which maps both to calibrated render_world.
That report also contains per-timestamp relative GT and learned states in
`joint_state_samples`. The existing `joint_axis_preview.png` shows Gaussian
centres with the axis lines; it is not a Gaussian splat render.

To evaluate an existing run without retraining:

```bash
freeart_target="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/freeart"
eval "$(/home/samuelemara/miniconda3/envs/freeartgs/bin/python -B utils/local_storage.py --outputs-root "$freeart_target")"
freeart_sources="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render"
/home/samuelemara/miniconda3/envs/freeartgs/bin/python -B evaluate/evaluate_local_joint_axis.py --scene "$freeart_sources/USB_100109" --dataset "$freeart_target/work/datasets/USB_100109_multiview" --results "$freeart_target/USB_100109_multiview/splatfacto-art"
```

Validate the full batch without writing files or starting training:

```bash
bash pipelines/train_local_datasets.sh --all-cameras --dry-run
```

Train a single object, still using all eight views:

```bash
bash pipelines/train_local_datasets.sh --object fridge_10905
```

Run a short compatibility check with eight sampled timestamps and ten iterations
per training stage:

```bash
bash pipelines/train_local_datasets.sh --object USB_100109 --iterations 10 --max-frames 8 --frame-stride 8
```

## Calibration and motion initialization

Raw datasets contain `rgb/cam_XXX`, `depth_npy/cam_XXX`, `mask/cam_XXX`,
`metadata/cameras.json`, and a COLMAP text model in `colmap/sparse/0`.
Depth arrays are metric float32 values; masks describe the whole object.

Intrinsics come from `cameras.txt`, matched to each camera using the camera IDs
in `images.txt`. `PINHOLE` and `SIMPLE_PINHOLE` are supported. Fixed camera poses
come from the COLMAP world-to-camera rotations/translations, inverted to
camera-to-world poses. Each observation retains its own intrinsics and pose.
The adapter rejects moving cameras and mismatched timestamp stems across views.

The existing AllTracker, two-part segmentation and ICP frontend first processes
each camera's temporal sequence. The adapter guesses each view's moving part
by its estimated motion, relabels masks with that heuristic, and uses the view with the
best initial-frame visibility and valid-pose coverage (breaking ties by the
largest difference between part motion scores) to initialize one shared motion
trajectory. COLMAP calibration places all views in the common coordinate frame.
The two Gaussian part models then train jointly across all observations, followed
by shared object-pose refinement, joint estimation, and articulated training.
Ground-truth joint annotations are not used for training. Segmentation and motion
initialization still depend on the existing frontend; calibration alone does not
guarantee their accuracy.
GT projection checks demonstrate that the labels are inconsistent when the
moving part is occluded; for example, storage cam_004 assigns many visible
static surfaces to motion. The frontend needs joint correspondence in calibrated
3D and must allow views with no moving pixels. This issue remains unresolved.

## Files and reruns

If an older run failed with a shortened `global_poses_*.npy`, retry its object
using completed tracking rather than starting the frontend again:

```bash
bash pipelines/train_local_datasets.sh --object fridge_10905 --reuse-tracking --evaluate-joint-axis
```

This regenerates configs, recomputes ICP for camera sequences with missing pose
slots, and restarts reconstruction. Missing prepared cameras go through the
frontend. Selected-object retries keep the unselected objects' batch summary records.
ICP now preserves RGB timestamp slots even when an invisible part has no PLY
file, records `pose_valid_*.npy`, and keeps original filenames in JSON exports.

The batch creates these paths relative to the target:

- `work/input/<object>_multiview_camXXX/`: converted per-camera RGB-D inputs.
- `work/datasets/<object>_multiview_camXXX/`: per-camera preprocessing and tracking.
- `work/datasets/<object>_multiview/`: merged calibrated observations, part transforms
  and `multiview.json`, which maps observations to shared timestamps.
- `work/configs/<object>_multiview.yaml`: complete shared training config.
- `<object>_multiview/`: shared part models, blended model and articulation.
- `<object>_multiview/splatfacto-art/`: final checkpoint, exported
  `articulation.json`, Gaussian PLY and `eval_results/` renders. GT evaluation adds
  `joint_axis_error.json` and `joint_axis_preview.png`; `--visualize-results` adds
  `comparison/motion_vs_gt.mp4`, `motion_vs_gt.gif` and comparison stills when GT
  exists, or `motion_axis_debug.mp4` / `.gif` without GT.
- `batch_logs/<object>_multiview.log` and `summary.json`: progress and outcomes.
- `.runtime/`: temporary files, Matplotlib/Torch/Hugging Face/CUDA caches and
  compiled extensions. Python bytecode writes to the repo/environment are disabled.

Old prepared inputs and datasets in the repo are not reused or migrated.
`--reuse-tracking` only reuses prepared data under the target's `work/datasets/`.
This avoids reading the truncated files left by the previous disk-full run.
Existing completed result folders and unselected summary rows are preserved.
Add `--skip-completed` to check the selected objects and leave completed runs
untouched. A job is skipped only if its summary entry has `status: passed` and
`training_completed: true`, its final checkpoint matches the requested iteration
count, and its Gaussian PLY, part weights, config, articulation and exported cameras
are present and nonempty. Exported JSON must parse successfully. Requested GT
evaluation requires a GT report; visualization requires the matching MP4/GIF
outputs, including estimated-only videos for scenes without GT. Visualization
failures are retried. This checks completion, not
reconstruction quality.

```bash
bash /home/samuelemara/FreeArtGS/pipelines/train_local_datasets.sh \
  --skip-completed --all-cameras --evaluate-joint-axis --visualize-results \
  --objects oven_101917 scissor_11100 washer_103776 blade_103706 \
    fridge_10905 stapler_103111 USB_100109
```

Skipped objects bypass raw-frame validation, preprocessing and training, preserving
their logs and summary records. If all selected objects are complete, the launcher
returns without requiring CUDA or creating working/cache directories. Use
`--dry-run` with `--skip-completed` to print what will run without writing files.
Incomplete objects still restart reconstruction; this flag does not resume
checkpoints. Without `--skip-completed`, the previous rerun behavior is preserved.

The first run with the new cache paths may download model weights and rebuild
CUDA extensions.
If a raw source folder moves, retry without `--reuse-tracking`. Converted inputs
with the same object/camera names can be rebuilt when their original source path
no longer exists; tracking is regenerated against the selected source. Reuse
still rejects a different source path to avoid using stale motion estimates.

The runner requires at least 10 GiB free on the target before training and checks
again between cameras and before multiview assembly. A disk-full error from
Python or a subprocess stops the batch; other training errors still allow later
objects to run. Batch summaries are replaced atomically, so a failed write does
not truncate the previous summary. Dry runs create no working/cache directories.

Smoke tests with frame limits/stride get separate names such as
`USB_100109_multiview_f8_s8`. Raw source datasets are preserved. Rerunning replaces
matching prepared inputs, configs and logs and starts new training; it does not
resume checkpoints. A failed object is recorded; later objects still run unless
the failure concerns storage capacity.

Generated configs use the source camera's far plane (10 m here), avoiding the
original 3 m depth cutoff. FreeArt-21-specific external evaluation is disabled
because these exports do not contain its annotation layout. Nerfstudio's own
validation remains enabled.
New local configs also set `quaternion_order: xyzw` to convert model rotations
to GSplat's WXYZ input convention. Existing checkpoint configs default to legacy
rendering. Corrected runs need fresh part reconstructions; switching convention
on an old final checkpoint does not repair its learned Gaussian geometry.
Newly generated configs retain the official tracking threshold of 2,000 rather
than the previous local override of 500. Reusing tracking does not recompute its
segmentation. Older failed retries also report CUDA out of memory; details
are in the official pipeline review.

The single-camera converter remains available for the original pipeline:

```bash
python samuele/convert_standard_to_freeart.py --input datasets/USB_100109 --camera cam_000 --output input/USB_100109_cam000 --write-config configs/USB_100109_cam000.yaml
```

Regression checks, including calibration, synchronization, shared motion and
per-view articulated-parser intrinsics:

```bash
/home/samuelemara/miniconda3/envs/freeartgs/bin/python -m unittest discover -s tests -v
```

Validation artifacts and GPU smoke logs are in `outputs/dataset_validation/`.
Local checks validated all 4,800 raw RGB-D-mask triplets in the ten selected objects
and passed 45 regression
tests. The eight-camera USB smoke run completed every reconstruction stage with
finite checkpoint parameters and logged losses, exporting eight shared object
poses, eight joint parameters and 64 per-view previews. Its report is
`multiview_training_checks.json`; the successful reconstruction log is
`multiview_reconstruction_gpu.log`.
Short smoke runs verify pipeline compatibility; they do not establish convergence.
