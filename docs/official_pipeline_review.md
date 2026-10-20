# Review against the official FreeArtGS release

Reviewed on 2026-10-09. The official repository is
https://github.com/FreeArtGS/FreeArtGS. The local base is its initial public
release, commit `061c6c8`.

The current eight-camera adapter is experimental. Completing training or
reporting a small axis angle does not establish correct part reconstruction.
The frontend has confirmed part-label errors under occlusion. Do not treat the
batch as a validated reproduction of the paper.

## What the authors document

The [official project](https://freeartgs.github.io/) takes a monocular RGB-D
video as input. The
[README](https://github.com/FreeArtGS/FreeArtGS#quick-start) documents:

```bash
export MPLBACKEND=Agg
bash pipelines/freeartgs.sh --config configs/freeart-21.yaml --nsobject_name 100109
```

The [official pipeline](https://github.com/FreeArtGS/FreeArtGS/blob/main/pipelines/freeartgs.sh)
runs preprocessing, temporal two-part tracking, ICP, separate part reconstruction,
moving-part selection, blending, joint estimation, articulated refinement and
benchmark evaluation. It does not document a calibrated eight-camera input mode.
The example object and evaluation assets belong to FreeArt-21; this command is
not directly a command for the local `datasets/USB_100109` layout.

The local adapter retains the reconstruction stages but changes their input and
motion initialization. Each camera is tracked independently. Part labels are
selected independently by estimated motion, then one camera supplies the shared
initial trajectory. Calibrated observations jointly train each part model and
the final model, with one pose/joint state per timestamp. Fixed calibrated camera
optimization is disabled. These are local changes, not author-provided multiview
support.

## Confirmed segmentation problem

An audit projects the raw first-frame `points3D_parts.ply` into each calibrated
view. Samples must be inside the image and within 2 mm of its observed depth;
occluded samples are discarded. GT part 0 is static, part 1 is moving, as recorded
in `metadata/colmap_export.json`. GT is used for this diagnostic only.

Examples from the currently prepared inputs:

| Object / camera | Visible GT samples | Moving GT samples | Static samples labeled moving |
| --- | ---: | ---: | ---: |
| storage / cam_004 | 1,096 | 0 | 745 |
| storage / cam_003 | 1,285 | 0 | 571 |
| blade / cam_007 | 61 | 0 | 60 |
| fridge / cam_002 | 948 | 67 | 600 |

These are sampled surface diagnostics, not dense mask IoU measurements. They
establish that the statement that the adapter relabels parts consistently across
views was incorrect. When the moving part is invisible, a forced two-part split
can divide the static body and assign one region to motion. Choosing the larger
ICP motion score does not resolve that ambiguity. Sharing timestamps and camera
calibration does not make those semantic labels correct.

Detailed counts: `outputs/dataset_validation/official_review_part_masks.json`.
The next required algorithmic correction is joint part correspondence and motion
initialization in calibrated 3D, allowing a view to contain no moving pixels.
GT part labels must remain evaluation inputs unless supervised training is
explicitly selected. Another long batch with the same independent segmentation
cannot establish that this issue is fixed.

## Confirmed Gaussian rotation convention mismatch

The local FreeArtGS loaders, world transforms, articulation transforms and PLY
export treat model quaternions as `XYZW`. The installed GSplat 0.1.11
(`gsplat/cuda/csrc/helpers.cuh` and `gsplat/_torch_impl.py`) interprets its input
as `WXYZ`. The model passed its internal quaternion directly to GSplat, including
the covariance rotation used for Gaussian splitting.

A CPU check rotates an anisotropic Gaussian by 90 degrees and compares the
resulting rendered covariance with `R Sigma R^T`. The original path has a
Frobenius discrepancy of 1.40985 in that synthetic example; converting component
order at the GSplat boundary restores the expected covariance. Evidence is in
`outputs/dataset_validation/official_review_quaternion_convention.json`.

The correction applies to part, blend and articulated rendering, Gaussian
splitting and the articulation viewer's PLY rendering. Newly generated local
configs explicitly select `quaternion_order: xyzw`. Existing checkpoints and
configs retain `legacy` behavior by default; changing their convention would
change their images. Existing artifacts were not re-exported or migrated.
Corrected training must start from corrected part reconstructions, rather than
resume an old final checkpoint with the new convention. The PLY viewer now uses
the quaternion order declared by the PLY format; legacy exports can still differ
from the original checkpoint's renders.

## Evaluation and visualization

The earlier local metric measured only
`degrees(acos(abs(dot(predicted_axis, gt_axis))))`, after calibrated camera
alignment. That formula is a direction diagnostic; it does not measure geometry,
part assignment, joint state or revolute pivot position. The
[official evaluator](https://github.com/FreeArtGS/FreeArtGS/blob/main/evaluate/evaluate_freeartgs.py)
also evaluates joint type/state/position, cameras and geometry, with optional
rendering evaluation. Its URDF and FreeArt-21 annotation layout differs from the
local render-world metadata, so its coordinate transforms cannot be copied
blindly onto these datasets.

The local evaluator now also reports type correctness, relative joint-state
error at matched source timestamps, revolute axis-line distance, camera position
consistency and observation counts. State error removes only the first-state
offset and applies the predicted axis sign; it does not fit a scale to GT.
It explicitly labels its output as local diagnostics rather than the official
benchmark. Geometry and held-out view quality are not assessed by this report.

The completed storage result has 480 exported views and 60 joint states:

- Axis direction error: 0.009873 degrees.
- Joint type: prismatic, matching GT.
- Relative displacement MAE: 0.5521 mm; maximum error: 1.3873 mm.
- Predicted displacement range: 0.532538 m; GT: 0.532000 m.
- Maximum camera position discrepancy: approximately 1.42 micrometres.
- Mean foreground PSNR of its 480 saved **training-view** splat renders:
  35.0522 dB. These are not held-out or novel-view results.

This storage result does fit the observed images and translation well, despite
poor initial part labels in some cameras. Its small angle cannot demonstrate
that the adapter works for the other objects.

A prismatic joint has no identifiable pivot location: only its translation
direction and states determine motion. Storage's exported origin is near the
first camera, about 3.47 m away from the annotated point along the sliding
direction. Drawing that as a recovered physical hinge is misleading. Its
axis-position error is now explicitly undefined.

The Open3D viewer displays Gaussian centres. Its soft animation interpolates
point displacement, whereas the trained model renders static and fully moving
Gaussian copies with weighted opacity. Even its PLY-based splat capture is an
approximation of the checkpoint's soft model. Use the saved training renders to
inspect the trained model's actual images; do not judge reconstruction from
the point-centre plot alone.

Comparison images and motion plot:

- `outputs/dataset_validation/storage_gt_render_comparison.png`
- `outputs/dataset_validation/storage_joint_motion.png`
- `outputs/dataset_validation/official_review_storage_renders.json`

## Why the four objects still failed

The original batch had shortened pose arrays when an empty part cloud was not
written as a PLY. That timestamp-slot bug was fixed previously.

The **latest retry logs** now show another cause: CUDA out of memory. USB reached
articulated training step 9,900 before failing. Its exception reports 14.19 MiB
free on a 7.86 GiB GPU, with other processes consuming 2.94 GiB and 4.50 GiB.
Blade, fridge and stapler failed in GPU ICP with Open3D out-of-memory exceptions.
The other process IDs in that historical exception are no longer present at
review time. This review did not terminate processes or launch training.

There are no completed final articulation exports for these four production
jobs. No GT error should be reported for them. Batch `status: passed` is retained
for compatibility but now explicitly means training completed, with
`reconstruction_quality_validated: false`.

## Changes and checks made in this review

- Extended local GT diagnostics without using GT to train the model.
- Corrected quaternion component order at the GSplat boundary for newly
  generated local runs, with legacy checkpoint compatibility.
- Restored the generated tracking config's official `min_trajectory_count: 2000`
  instead of the local override of 500. This config correction is not a remedy
  for multiview segmentation; existing tracking artifacts were not retrained.
- Marked the adapter and its manifests as experimental and unvalidated for
  part consistency; removed the claim of consistent relabeling.
- Compared all 480 saved storage training renders to their matching inputs.
- Ran 27 CPU regression tests successfully, including covariance rotation,
  legacy quaternion order, sampled source-timestamp
  mapping, axis-sign handling, state offset, type mismatch and missing states.

No new production training, corrected GPU rendering, GPU ICP, GUI session or
novel-view evaluation was performed. The segmentation error remains unresolved
in the training frontend.
