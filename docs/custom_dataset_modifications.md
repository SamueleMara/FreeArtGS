# FreeArtGS custom dataset modifications and files to gather

Prepared on 2026-10-15. This describes the current custom dataset adapter and
output-location changes, including the fixes made during the failed batch runs.

## What to gather

The source bundle is `/tmp/freeartgs_custom_dataset_changes.tar.gz`.
It contains the changed source files, tests, documentation, the unchanged YAML
template required by the converter, and `BASE_COMMITS.txt` recording both Git
revisions. The exact repository-relative file list is
[`custom_dataset_files.txt`](custom_dataset_files.txt).

The archive contains a `FreeArtGS/` directory. It is an overlay for an existing
checkout of the same base repository with its Nerfstudio submodule initialized,
not a complete standalone installation. The installed dependencies, model
weights, raw datasets, trained checkpoints and rendered results are not bundled.
The archive is in `/tmp`; copy/download it before temporary files are cleaned.

Several added files are currently untracked, and the Nerfstudio changes are
inside a Git submodule. A parent-repository `git diff` alone misses these files.
Transfer the complete bundle or every path in the manifest.

## Dataset conversion and calibration

Extended the existing `samuele/convert_standard_to_freeart.py`:

- Consume RGB, metric depth and foreground masks from the per-camera folders.
- Take intrinsics from `colmap/sparse/0/cameras.txt`, matching camera IDs through
  `images.txt`; support PINHOLE and SIMPLE_PINHOLE and reject unsupported models.
- Normalize observations into sequential five-digit PNGs, depth NPZ files and
  `cam_K.txt`, retaining original camera/frame provenance in metadata.
- Validate matching filenames, dimensions and valid foreground depth before
  writing; protect raw source directories from replacement.
- Generate configs from `configs/freeart-21.yaml`, using the source depth range,
  supplied foreground masks and the corrected quaternion convention.
- Disable the FreeArt-21-specific data split and external benchmark evaluator
  because these datasets do not have that benchmark's annotation layout.
- Report unavailable input datasets explicitly instead of a misleading ZIP error.

Changed `utils/utils.py` to validate intrinsics, RGB/depth/mask/feature alignment,
array shapes and finite values. Changed `preprocessing/segment.py` to reject
identical input/output directories and import Grounded-SAM only when required.

## Shared calibrated multiview training

Added `preprocessing/prepare_multiview.py`:

- Invert calibrated COLMAP world-to-camera poses into camera-to-world poses.
- Require fixed cameras and synchronized source timestamps.
- Merge eight camera sequences into 480 observations sharing 60 timestamps.
- Preserve per-observation intrinsics and calibrated camera poses.
- Write `multiview.json` linking observations to cameras, source frames and time.
- Use canonical part IDs 0 = static and 1 = moving.
- Select moving parts from estimated motion and use a reference camera to
  initialize a shared motion trajectory in the calibrated coordinate system.

Changed these files in `reconstruction/nerfstudio/`:

| File relative to the submodule | Change |
| --- | --- |
| `nerfstudio/data/utils/dataparsers_utils.py` | Load and validate multiview observation metadata. |
| `nerfstudio/data/dataparsers/twopart_blend_dataparser.py` | Preserve per-view intrinsics and supply shared timestamp indices. |
| `nerfstudio/data/dataparsers/splatfacto_art_dataparser.py` | Use one joint parameter per shared timestamp rather than per camera observation. |
| `nerfstudio/models/twopart_blend.py` | Share object poses across simultaneous views and validate their initialization. |
| `nerfstudio/models/splatfacto.py` | Convert internal XYZW rotations to GSplat WXYZ when selected; clamp initial spacing and color logits to prevent nonfinite parameters. |
| `nerfstudio/models/splatfacto_art.py` | Use corrected quaternion conversion and unique observation filenames for saved renders. |

Existing configs retain the default legacy quaternion behavior. Newly generated
local configs explicitly select `quaternion_order: xyzw`. Correcting an old final
checkpoint's config alone does not repair geometry learned with the old convention.

## ICP and Gaussian initialization fixes

- `icp/color_icp_gpu.py`: retain timestamp slots for missing/empty part clouds,
  save `pose_valid_*.npy`, and preserve original image filenames in transforms.
- `reconstruction/generate_ply.py`: deduplicate multiview seed points to avoid
  zero neighbour distances during Gaussian scale initialization.

## Launching, selection and reruns

Added `pipelines/train_local_datasets.py` and
`pipelines/train_local_datasets.sh`:

- Select exact objects with `--object` or `--objects`; exclude episode 000 and
  generated datasets from discovery.
- Validate selected raw sequences before training and train objects sequentially,
  using all camera observations in a shared model per object.
- Support frame limits, frame stride and iteration limits for compatibility runs.
- Reuse matching prepared tracking with `--reuse-tracking`, repairing short ICP
  pose sequences when required.
- Add `--skip-completed`: require a successful training summary, the final
  checkpoint matching the requested iteration count, nonempty exports and valid
  JSON. Also require GT reports and video outputs when requested. Preserve
  skipped logs and summary records; return without CUDA when all jobs are skipped.
- Stop after source disappearance or storage failures; retain per-object errors
  and continue after other training failures.
- Replace summaries atomically so failed writes do not truncate earlier results.
- Allow rebuilding owned converted inputs after their original source directory
  disappears and the same object/camera is selected at its new location. Changed
  source paths are not accepted for tracking reuse.

Changed `pipelines/freeartgs.sh` to recognize the multiview configuration, avoid
running ICP again on the merged dataset, use canonical moving part 1 and honor
the external evaluation enable/disable setting.

Changed `pipelines/train_parts.sh`, `pipelines/twopart_blend.sh` and
`pipelines/splatfacto_art.sh` to pass camera/quaternion settings and cache images
in CPU RAM. The calibrated batch disables camera optimization. Model training
continues to use CUDA; CPU image caching is not CPU model training.

## GT evaluation and visualization

Added `evaluate/evaluate_local_joint_axis.py`:

- Align learned axes/origins with calibrated `render_world` using exported cameras.
- Read GT from `pointcloud/joint_metadata.json` and `metadata/frame_values.csv`.
- Report unsigned axis direction error, joint type, relative joint-state errors,
  revolute axis-line distance and camera consistency.
- Match original source timestamps, remove the initial state offset and handle
  axis sign without fitting a scale to GT.
- Save the numeric report and a Gaussian-centre plot with predicted/GT axes.

Added `evaluate/visualize_local_comparison.py` to produce all-camera MP4/GIF
comparisons of saved Gaussian renders against GT RGB, with projected joint axes.
GT foreground masks composite the reference images onto white. Prismatic arrows
share a display anchor because a prismatic pivot location is unidentifiable.

Changed `reconstruction/visualize_articulation.py` to read exported RGB colors,
correct quaternion ordering for PLY splat rendering, overlay GT axes and preserve
the learned origin. Added `pipelines/view_local_articulation.sh` for its environment.

GT joint annotations remain evaluation/diagnostic inputs, not training targets.
Comparison videos use saved training-view renders. The interactive PLY animation
is an approximation of the checkpoint's soft part mixture.

## All batch-generated files under the output target

Added `utils/local_storage.py` and configured the batch launcher so the target is
the root for working data, final results, logs, temporary files and library caches.

Default raw source, read only:

```text
/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render
```

Default target, overridable with `--outputs-root`:

```text
/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/freeart/
├── work/
│   ├── input/       # Converted camera inputs
│   ├── datasets/    # Features, tracking, ICP and merged observations
│   └── configs/     # Generated YAML configs
├── .runtime/        # Temporary files and library/compiler caches
├── batch_logs/
└── <object>_multiview/
    ├── part_0/
    ├── part_1/
    ├── twopart_blend/
    └── splatfacto-art/
```

Specifically:

- Separate `--datasets` raw sources from generated working datasets.
- Redirect `TMPDIR`, `TEMP`, `TMP` and Python temporary directories.
- Redirect XDG cache/config/data, Matplotlib, Torch Hub, Torch extensions,
  TorchInductor, Hugging Face, Transformers, CUDA, Triton and Numba caches.
- Disable Python bytecode writes to the repository/environment.
- Check working/cache directories resolve inside the selected target.
- Require 10 GiB free on the target and repeat checks between cameras and before
  assembling the multiview dataset; stop after disk-full/quota errors.
- Activate the installed `freeartgs` Conda environment and CUDA 12.8. `MAX_JOBS=2`
  limits extension compilation parallelism, not model training to two CPU threads.
- Delay runtime directory creation until the batch actually has jobs to run;
  dry runs and fully skipped batches create no work/cache directories.

This routing applies to the local batch launcher and its child processes.
Standalone upstream commands and standalone tools require matching output paths
and runtime environment settings; this is not an OS filesystem sandbox.
Source code/documentation changes and earlier local artifacts are not migrated
automatically. The source bundle described above is a separate temporary artifact.

## Tests, documentation and historical generated files

Gather all six test files listed in the manifest. They cover dataset loading,
calibration, multiview timestamps, Gaussian rotations, ICP alignment, GT metrics,
runtime storage, full-disk handling, source relocation, skipping completed jobs
and visual comparison helpers. Last verified: 45 regression tests passed.

Also gather `README.md`, `.gitignore`, `docs/local_datasets.md` and
`docs/official_pipeline_review.md`. `/input/` was added to `.gitignore`.

Historical generated YAMLs under `configs/local_datasets/` and the early
`configs/*_cam000.yaml` files are not required to transfer these changes. The
current runner regenerates configs under the target's `work/configs/`. The bundle
does include the unchanged `configs/freeart-21.yaml` template it reads.

If you also need trained results, gather them separately from the SSD target:

- `<object>_multiview/` for complete part/blend/articulated models and checkpoints.
- `batch_logs/` for run outcomes and diagnostics.
- `work/configs/` for exact generated training settings.
- `work/datasets/` for prepared observations and manifests referenced by saved
  model configs, if reproducing renders or resuming inspection elsewhere.
- The raw sources, if recomputing tracking or GT comparisons on another machine.

Copied configs can contain absolute paths; these need updating on another machine.

## Remaining scientific limitation

The multiview frontend still segments cameras independently and initializes
shared motion from one camera. Part-label errors under occlusion remain
unresolved. Completed training and a small axis angle do not establish correct
geometry, part assignment or held-out rendering quality.

This document and bundle do not represent a new GPU training or quality-validation run.
