#!/usr/bin/env python3
"""
Convert a multi-camera episode dataset to a FreeArtGS-compatible monocular RGB-D object folder.

Input standard layout, either as an extracted directory or a .zip:

    episode_000000/
    ├── rgb/<camera>/frame_000000.jpg|png|...
    ├── depth_npy/<camera>/frame_000000.npy       # preferred, depth in meters
    ├── depth_png/<camera>/frame_000000.png       # optional fallback, scaled by --depth-png-scale
    ├── mask/<camera>/frame_000000.png
    └── metadata/cameras.json

Output FreeArtGS-style layout:

    <output-root>/
    ├── rgb/00000.png
    ├── depth/00000.npz      # contains key: depth, float32 meters
    ├── masks/00000.png      # binary 0/255 mask
    ├── cam_K.txt            # 3x3 intrinsics
    └── metadata/freeart_conversion.json

Example:

    python convert_standard_to_freeart.py \
        --input episode_000000.zip \
        --output examples/real_cam000 \
        --camera cam_000 \
        --overwrite

Then use a FreeArtGS config with:

    paths:
      object_name: real_cam000
      original_data_dir: examples
      dataset_root: datasets
      output_root: outputs

and set preprocess.skip_segment: true.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import numpy as np
from PIL import Image
import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEPTH_EXTS = {".npy", ".npz", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class FrameTriplet:
    stem: str
    rgb_path: Path
    depth_path: Path
    mask_path: Path | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one selected camera from a multi-camera standard dataset to FreeArtGS format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input episode directory or .zip file. The scene root must contain rgb/, depth_npy or depth_png, mask/, metadata/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output FreeArtGS object directory, e.g. examples/my_real_object.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera to select, e.g. cam_000. Also accepts source_camera_name if present in cameras.json.",
    )
    parser.add_argument(
        "--depth-source",
        choices=["auto", "npy", "png"],
        default="auto",
        help="Depth source. auto prefers depth_npy/<camera>, then depth_png/<camera>.",
    )
    parser.add_argument(
        "--depth-png-scale",
        type=float,
        default=0.001,
        help="Meters per integer unit when reading depth_png. Ignored for depth_npy. Default: 0.001.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth matched frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap after frame-stride.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Output numbering start index. Default: 0 -> 00000.png/npz.",
    )
    parser.add_argument(
        "--name-width",
        type=int,
        default=5,
        help="Zero-padding width for output frame names. Default: 5.",
    )
    parser.add_argument(
        "--rgb-format",
        choices=["png", "copy"],
        default="png",
        help="png converts all RGB frames to .png. copy preserves original RGB file extension.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="Threshold for binarizing masks to 0/255.",
    )
    parser.add_argument(
        "--allow-missing-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If masks are missing, create all-white masks. Useful only when you know RGB is already object-cropped.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.0,
        help="Depth values <= this value are written as 0. Default: 0.0.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Depth values > this value are written as 0. Default: keep all finite positive values.",
    )
    parser.add_argument(
        "--write-config",
        type=Path,
        default=None,
        help="Optional path to write a complete FreeArtGS pipeline YAML config for this object.",
    )
    parser.add_argument(
        "--original-data-dir",
        type=str,
        default=None,
        help="Value for paths.original_data_dir in the optional YAML config. Default: parent of --output.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets",
        help="Value for paths.dataset_root in the optional YAML config.",
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default="outputs",
        help="Value for paths.output_root in the optional YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be converted; do not write files.",
    )
    return parser.parse_args(argv)


def natural_key(path_or_str: Path | str) -> list[object]:
    s = Path(path_or_str).stem if isinstance(path_or_str, Path) else str(path_or_str)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def extract_if_zip(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Return an extracted/root directory and a tempdir handle that must stay alive."""
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset is missing or disconnected: {input_path}")
    if input_path.is_dir():
        return input_path, None
    if input_path.suffix.lower() != ".zip":
        raise ValueError(f"--input must be a directory or .zip file, got: {input_path}")
    tmp = tempfile.TemporaryDirectory(prefix="freeart_convert_")
    print(f"[INFO] Extracting zip to temporary directory: {tmp.name}")
    with ZipFile(input_path, "r") as zf:
        zf.extractall(tmp.name)
    return Path(tmp.name), tmp


def find_scene_root(root: Path) -> Path:
    """Find the directory containing rgb/, metadata/, etc."""
    root = root.resolve()
    if (root / "rgb").is_dir():
        return root
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "rgb").is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(str(p) for p in candidates)
        raise RuntimeError(f"Multiple candidate scene roots found; pass one directly: {names}")
    raise FileNotFoundError(f"Could not find a scene root containing rgb/ under: {root}")


def list_cameras(scene_root: Path) -> list[str]:
    rgb_root = scene_root / "rgb"
    return sorted([p.name for p in rgb_root.iterdir() if p.is_dir()])


def load_colmap_intrinsics(scene_root: Path, camera_name: str) -> dict | None:
    sparse = scene_root / "colmap" / "sparse" / "0"
    cameras_path = sparse / "cameras.txt"
    if not cameras_path.exists():
        return None
    camera_ids = set()
    # COLMAP images.txt alternates image records and their 2D observations.
    with (sparse / "images.txt").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError(f"Invalid COLMAP image record in {sparse / 'images.txt'}")
            if Path(fields[9].strip()).parts[0] == camera_name:
                camera_ids.add(int(fields[8]))
            next(handle, None)
    if len(camera_ids) != 1:
        raise ValueError(f"Expected one COLMAP camera ID for {camera_name}, found {sorted(camera_ids)}")
    camera_id = camera_ids.pop()
    for line in cameras_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if int(fields[0]) != camera_id:
            continue
        model = fields[1]
        params = [float(p) for p in fields[4:]]
        if model == "PINHOLE" and len(params) == 4:
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE" and len(params) == 3:
            fx, cx, cy = params
            fy = fx
        else:
            raise ValueError(f"Unsupported COLMAP camera model {model}; undistorted pinhole RGB-D is required")
        return dict(fx=fx, fy=fy, cx=cx, cy=cy, width=int(fields[2]), height=int(fields[3]),
                    intrinsics_source=str(cameras_path), colmap_camera_id=camera_id)
    raise ValueError(f"COLMAP camera ID {camera_id} not found in {cameras_path}")


def load_camera_entry(scene_root: Path, camera_name_or_source: str) -> dict:
    meta_path = scene_root / "metadata" / "cameras.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Camera metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cameras = meta.get("cameras", [])
    for cam in cameras:
        if cam.get("name") == camera_name_or_source or cam.get("source_camera_name") == camera_name_or_source:
            cam = dict(cam)
            colmap = load_colmap_intrinsics(scene_root, cam["name"])
            if colmap is not None:
                cam.update(colmap)
            return cam
    available = [f"{c.get('name')} (source={c.get('source_camera_name')})" for c in cameras]
    raise KeyError(
        f"Camera '{camera_name_or_source}' not found in {meta_path}. Available: {', '.join(available)}"
    )


def collect_files(folder: Path, exts: set[str]) -> dict[str, Path]:
    if not folder.exists():
        return {}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return {p.stem: p for p in files}


def choose_depth_folder(scene_root: Path, camera: str, depth_source: str) -> Path:
    if depth_source == "npy":
        folder = scene_root / "depth_npy" / camera
        if not folder.is_dir():
            raise FileNotFoundError(f"Requested --depth-source npy but folder is missing: {folder}")
        return folder
    if depth_source == "png":
        folder = scene_root / "depth_png" / camera
        if not folder.is_dir():
            raise FileNotFoundError(f"Requested --depth-source png but folder is missing: {folder}")
        return folder

    # auto
    for candidate in [scene_root / "depth_npy" / camera, scene_root / "depth" / camera, scene_root / "depth_png" / camera]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No depth folder found for {camera}. Expected one of depth_npy/{camera}, depth/{camera}, depth_png/{camera}."
    )


def build_triplets(scene_root: Path, camera: str, depth_source: str, allow_missing_mask: bool) -> tuple[list[FrameTriplet], Path]:
    rgb_dir = scene_root / "rgb" / camera
    if not rgb_dir.is_dir():
        cams = list_cameras(scene_root)
        raise FileNotFoundError(f"RGB folder not found: {rgb_dir}. Available cameras: {cams}")

    depth_dir = choose_depth_folder(scene_root, camera, depth_source)
    mask_dir = scene_root / "mask" / camera
    if not mask_dir.is_dir():
        mask_dir = scene_root / "masks" / camera

    rgb_by_stem = collect_files(rgb_dir, IMAGE_EXTS)
    depth_by_stem = collect_files(depth_dir, DEPTH_EXTS)
    mask_by_stem = collect_files(mask_dir, IMAGE_EXTS)

    if set(rgb_by_stem) != set(depth_by_stem):
        raise ValueError(f"RGB/depth frame stems differ for {camera}: "
                         f"missing depth={sorted(set(rgb_by_stem) - set(depth_by_stem))[:10]}, "
                         f"missing RGB={sorted(set(depth_by_stem) - set(rgb_by_stem))[:10]}")

    common = sorted(set(rgb_by_stem) & set(depth_by_stem), key=natural_key)
    if not common:
        raise RuntimeError(
            f"No matching RGB/depth frame stems for camera {camera}.\n"
            f"RGB folder: {rgb_dir} ({len(rgb_by_stem)} files)\n"
            f"Depth folder: {depth_dir} ({len(depth_by_stem)} files)"
        )

    triplets: list[FrameTriplet] = []
    missing_masks = []
    for stem in common:
        mask_path = mask_by_stem.get(stem)
        if mask_path is None and not allow_missing_mask:
            missing_masks.append(stem)
            continue
        triplets.append(FrameTriplet(stem, rgb_by_stem[stem], depth_by_stem[stem], mask_path))

    if missing_masks and not allow_missing_mask:
        preview = ", ".join(missing_masks[:10])
        raise RuntimeError(
            f"Missing masks for {len(missing_masks)} matched RGB/depth frames in {mask_dir}. "
            f"Examples: {preview}. Use --allow-missing-mask to create white masks."
        )

    return triplets, depth_dir


def read_depth(path: Path, depth_png_scale: float) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as obj:
            if "depth" in obj.files:
                depth = obj["depth"]
            elif len(obj.files) == 1:
                depth = obj[obj.files[0]]
            else:
                raise KeyError(f"Depth npz has no 'depth' key and multiple arrays: {path}, keys={obj.files}")
    elif suffix in {".png", ".tif", ".tiff"}:
        depth = np.asarray(Image.open(path)).astype(np.float32) * float(depth_png_scale)
    else:
        raise ValueError(f"Unsupported depth file: {path}")

    depth = np.asarray(depth)
    if depth.ndim == 3:
        if depth.shape[-1] != 1:
            raise ValueError(f"Expected single-channel depth: {path}, shape={depth.shape}")
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth: {path}, shape={depth.shape}")
    return depth.astype(np.float32, copy=False)


def sanitize_depth(depth: np.ndarray, min_depth: float, max_depth: float | None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).copy()
    bad = ~np.isfinite(depth) | (depth <= float(min_depth))
    if max_depth is not None:
        bad |= depth > float(max_depth)
    depth[bad] = 0.0
    return depth


def save_rgb(src: Path, dst: Path, rgb_format: str) -> None:
    if rgb_format == "copy":
        shutil.copy2(src, dst)
        return
    img = Image.open(src).convert("RGB")
    img.save(dst)


def save_mask(src: Path | None, dst: Path, size_wh: tuple[int, int], threshold: int) -> None:
    if src is None:
        mask = np.full((size_wh[1], size_wh[0]), 255, dtype=np.uint8)
    else:
        m = Image.open(src).convert("L")
        if m.size != size_wh:
            raise ValueError(f"Mask size mismatch for {src}: mask={m.size}, rgb={size_wh}")
        mask = (np.asarray(m) > int(threshold)).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(dst)


def camera_intrinsics(camera_entry: dict) -> np.ndarray:
    camera_entry = dict(camera_entry)
    required = ["fx", "fy", "cx", "cy"]
    missing = [k for k in required if k not in camera_entry or camera_entry[k] is None]
    if missing:
        height, width = float(camera_entry["height"]), float(camera_entry["width"])
        fov = camera_entry.get("fov_y_deg")
        if fov is None or not 0 < float(fov) < 180 or min(height, width) <= 0:
            raise ValueError(f"Missing intrinsics {missing}; valid width, height and fov_y_deg are required")
        focal = height / (2 * np.tan(np.deg2rad(float(fov)) / 2))
        defaults = {"fx": focal, "fy": focal, "cx": width / 2, "cy": height / 2}
        for key in missing:
            camera_entry[key] = defaults[key]
    k = np.array(
        [
            [float(camera_entry["fx"]), 0.0, float(camera_entry["cx"])],
            [0.0, float(camera_entry["fy"]), float(camera_entry["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError(f"Invalid camera intrinsics: {camera_entry}")
    return k


def write_cam_k(out_path: Path, camera_entry: dict) -> np.ndarray:
    k = camera_intrinsics(camera_entry)
    np.savetxt(out_path, k, fmt="%.10f")
    return k


def maybe_write_config(args: argparse.Namespace, object_name: str, camera_entry: dict) -> None:
    if args.write_config is None:
        return
    original_data_dir = args.original_data_dir
    if original_data_dir is None:
        original_data_dir = str(args.output.resolve().parent)

    template = Path(__file__).resolve().parents[1] / "configs/freeart-21.yaml"
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["paths"] = dict(object_name=object_name, original_data_dir=original_data_dir,
                           dataset_root=args.dataset_root, output_root=args.outputs_root)
    config["split_eval"]["enabled"] = False
    # These inputs do not contain FreeArt-21's evaluation annotations.
    config["evaluation"] = {"enabled": False}
    max_depth = float(camera_entry.get("far") or 20.0)
    config["preprocess"].update(segment_prompt="object", skip_segment=True, pcd_max_depth=max_depth)
    config["tracking"].update(min_depth=0.01, max_depth=max_depth)
    for stage in ("train_parts", "twopart_blend", "splatfacto_art"):
        config[stage]["quaternion_order"] = "xyzw"
    args.write_config.parent.mkdir(parents=True, exist_ok=True)
    args.write_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"[OK] Wrote config template: {args.write_config}")


def convert(args: argparse.Namespace) -> None:
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.name_width != 5 or args.rgb_format != "png":
        raise ValueError("FreeArtGS requires five-digit PNG frame names (--name-width 5 --rgb-format png)")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be >= 1")
    if args.start_index != 0:
        raise ValueError("FreeArtGS requires sequential frame names starting at zero")
    if args.depth_png_scale <= 0 or not np.isfinite(args.depth_png_scale):
        raise ValueError("--depth-png-scale must be finite and positive")

    extracted_root, tmp = extract_if_zip(args.input)
    try:
        scene_root = find_scene_root(extracted_root)
        camera_entry = load_camera_entry(scene_root, args.camera)
        camera = str(camera_entry["name"])

        triplets, depth_dir = build_triplets(scene_root, camera, args.depth_source, args.allow_missing_mask)
        triplets = triplets[:: args.frame_stride]
        if args.max_frames is not None:
            triplets = triplets[: args.max_frames]
        if not triplets:
            raise RuntimeError("No frames left after applying --frame-stride/--max-frames.")

        out_root = args.output.expanduser().resolve()
        if out_root == scene_root or out_root in scene_root.parents:
            raise ValueError("Output must not replace the input scene or its parent")
        k = camera_intrinsics(camera_entry)
        # Validate the whole selected sequence before replacing any existing output.
        for tr in triplets:
            with Image.open(tr.rgb_path) as img:
                img.load()
                size_wh = img.size
            expected_size = (int(camera_entry["width"]), int(camera_entry["height"]))
            if size_wh != expected_size:
                raise ValueError(f"RGB/metadata size mismatch for {tr.rgb_path}: {size_wh} != {expected_size}")
            depth = sanitize_depth(read_depth(tr.depth_path, args.depth_png_scale), args.min_depth, args.max_depth)
            if depth.shape != (size_wh[1], size_wh[0]):
                raise ValueError(f"RGB/depth size mismatch for {tr.depth_path}: {depth.shape}")
            if tr.mask_path is None:
                foreground = np.ones(depth.shape, dtype=bool)
            else:
                with Image.open(tr.mask_path) as mask:
                    if mask.size != size_wh:
                        raise ValueError(f"RGB/mask size mismatch for {tr.mask_path}")
                    foreground = np.asarray(mask.convert("L")) > args.mask_threshold
            if not np.any(foreground & (depth > 0)):
                raise ValueError(f"No valid foreground depth for {tr.stem}; check masks and depth limits")
        if out_root.exists():
            if not args.overwrite and not args.dry_run:
                raise FileExistsError(f"Output already exists: {out_root}. Use --overwrite to replace it.")
            if not args.dry_run:
                shutil.rmtree(out_root)

        print(f"[INFO] Scene root: {scene_root}")
        print(f"[INFO] Selected camera: {camera} (source={camera_entry.get('source_camera_name')})")
        print(f"[INFO] RGB:   {scene_root / 'rgb' / camera}")
        print(f"[INFO] Depth: {depth_dir}")
        print(f"[INFO] Masks: {scene_root / 'mask' / camera}")
        print(f"[INFO] Output: {out_root}")
        print(f"[INFO] Frames to convert: {len(triplets)}")

        if args.dry_run:
            print("[DRY-RUN] Calibration and every selected RGB-D-mask frame validated. No files written.")
            return

        rgb_out = out_root / "rgb"
        depth_out = out_root / "depth"
        masks_out = out_root / "masks"
        meta_out = out_root / "metadata"
        for d in [rgb_out, depth_out, masks_out, meta_out]:
            d.mkdir(parents=True, exist_ok=True)

        write_cam_k(out_root / "cam_K.txt", camera_entry)

        conversion_meta = {
            "input": str(args.input),
            "scene_root": str(scene_root),
            "selected_camera": camera,
            "selected_source_camera_name": camera_entry.get("source_camera_name"),
            "num_frames": len(triplets),
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "output_format": {
                "rgb": "rgb/<index>.png" if args.rgb_format == "png" else "rgb/<index>.<source_ext>",
                "depth": "depth/<index>.npz with key 'depth' in meters",
                "masks": "masks/<index>.png binary 0/255",
                "intrinsics": "cam_K.txt 3x3",
            },
            "camera_entry": camera_entry,
            "K": k.tolist(),
            "frame_map": [],
        }

        for i, tr in enumerate(triplets):
            out_stem = f"{args.start_index + i:0{args.name_width}d}"
            rgb_suffix = tr.rgb_path.suffix.lower() if args.rgb_format == "copy" else ".png"
            rgb_dst = rgb_out / f"{out_stem}{rgb_suffix}"
            depth_dst = depth_out / f"{out_stem}.npz"
            mask_dst = masks_out / f"{out_stem}.png"

            with Image.open(tr.rgb_path) as img:
                size_wh = img.size
            save_rgb(tr.rgb_path, rgb_dst, args.rgb_format)

            depth = sanitize_depth(read_depth(tr.depth_path, args.depth_png_scale), args.min_depth, args.max_depth)
            if depth.shape[:2] != (size_wh[1], size_wh[0]):
                raise ValueError(
                    f"Depth size mismatch for {tr.depth_path}: depth={depth.shape[:2]}, rgb={(size_wh[1], size_wh[0])}"
                )
            np.savez(depth_dst, depth=depth.astype(np.float32, copy=False))

            save_mask(tr.mask_path, mask_dst, size_wh, args.mask_threshold)

            conversion_meta["frame_map"].append(
                {
                    "output_stem": out_stem,
                    "source_stem": tr.stem,
                    "rgb": str(tr.rgb_path),
                    "depth": str(tr.depth_path),
                    "mask": str(tr.mask_path) if tr.mask_path else None,
                }
            )

            if (i + 1) % 50 == 0 or (i + 1) == len(triplets):
                print(f"[INFO] Converted {i + 1}/{len(triplets)} frames")

        (meta_out / "freeart_conversion.json").write_text(json.dumps(conversion_meta, indent=2), encoding="utf-8")
        maybe_write_config(args, out_root.name, camera_entry)

        print("[OK] FreeArtGS-compatible object folder written.")
        print(f"     {out_root}")
        print("[NEXT] In FreeArtGS, set paths.object_name to this folder name and preprocess.skip_segment: true.")

    finally:
        # Keep tempdir alive during conversion, then remove it automatically.
        if tmp is not None:
            tmp.cleanup()


def main() -> None:
    try:
        convert(parse_args())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
