# Full-resolution masked real endpoints

After real conversion and mask generation, `create_standard_masks.py` calls
`colmap_masking.apply_scene_masks`. Scenes with `metadata/real_conversion.json`
now build their masked endpoints directly from saved registered metric depth.
Synthetic and HoI mask export retain their existing behavior.

With fraction 1, each finite positive `depth_npy` pixel whose source mask is
foreground becomes one COLMAP point, with RGB from that exact pixel and the
corresponding calibrated endpoint camera pose. Use `--point-fraction 0.125` for
one eighth, `0.25` for one quarter, or `1` for all points. Sampling selects
`floor(N * fraction)` pixels per view, at least one for nonempty views, evenly
across the valid pixel list. There is no voxel averaging. Points
from different cameras remain separate observations; cross-camera matches are
not fabricated. Initial models contain frame zero from all cameras; final models
contain the final synchronized frame from all cameras.

Outputs for each endpoint:

- `colmap/sparse/0` and `colmap_final/sparse/0`: sampled masked models and `pointcloud.ply`.
- Each endpoint's `sparse_unmasked/0`: preserved original seed model; its
  `pointcloud.ply` is refreshed from that model's actual points.
- `pointcloud/points3D.ply` and `points3D_final.ply`: match masked models.
- `pointcloud/points3D_unmasked.ply` and `points3D_final_unmasked.ply`:
  match preserved unmasked models, which may still be sampled/voxel merged.

PLY exports contain XYZ and RGB, without fabricated zero normals. Summary JSON
records per-camera foreground depth counts, mask hashes, and sampling settings.
Both endpoints must validate before publication starts. This does not refine
camera calibration or invent depth for missing measurements.

Rebuild all six current real scenes using existing RGB/depth/masks, without
rerunning conversion or segmentation:

```bash
REAL_ROOT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
for SCENE in "$REAL_ROOT"/*_20261021_161154_episode_00000[0-5]; do
  /home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
    /home/samuelemara/FreeArtGS/samuele/colmap_masking.py \
    --scene "$SCENE" --point-fraction 1 || break
done
```

This CPU-only command replaces masked models and the endpoint PLY exports inside
the selected scene directories. It preserves RGB, depth, masks, and unmasked
COLMAP camera/image/point model files. New batch runs use this path automatically
after mask generation.

The same flag is accepted by `convert_real_recording.py`,
`prepare_real_dataset_batch.py`, and `create_standard_masks.py`. On conversion
it samples the full valid pixel list before unmasked seed voxel fusion, replacing
the legacy stride grid. Masked exports apply the fraction after selecting
foreground. Rebuilds using existing scenes preserve the original unmasked seed
models and per-view clouds, and save the new fraction for future masked rebuilds.
Without an explicit setting, existing masked rebuilds use the saved fraction,
or 1 for older recordings. Raw RGB/depth images and masks are never downsampled.
