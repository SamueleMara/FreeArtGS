"""Use the shared SAPIEN mask exporter for host-side real-data preparation."""
import os
import sys
from pathlib import Path


def apply_scene_masks(scene):
    repo = Path(os.environ.get('SAPIEN_REPO', Path.home() / 'Joint3DGS/Sapien'))
    script_dir = repo / 'scripts'
    if not (script_dir / 'mask_colmap_models.py').is_file():
        raise FileNotFoundError(f'Missing shared mask exporter under {script_dir}; set SAPIEN_REPO to the Sapien checkout')
    sys.path.insert(0, str(script_dir))
    from mask_colmap_models import apply_scene
    return apply_scene(scene)
