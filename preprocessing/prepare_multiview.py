"""Combine calibrated fixed-camera sequences for joint FreeArtGS training."""
import json
import shutil
from pathlib import Path

import numpy as np

from samuele.convert_standard_to_freeart import camera_intrinsics, load_camera_entry
from nerfstudio.data.utils.colmap_parsing_utils import read_images_text
from preprocessing.compare_cam_motion import compute_metrics


def calibration_poses(scene):
    images_path = scene / "colmap/sparse/0/images.txt"
    poses = {}
    for image in read_images_text(images_path).values():
        camera = Path(image.name).parts[0]
        w2c = np.eye(4)
        w2c[:3, :3] = image.qvec2rotmat()
        w2c[:3, 3] = image.tvec
        c2w = np.linalg.inv(w2c)
        if camera in poses and not np.allclose(poses[camera], c2w, atol=1e-6):
            raise ValueError(f"Camera {camera} is moving; this adapter requires fixed calibrated cameras")
        poses[camera] = c2w
    return poses


def motion_score(poses):
    metrics = compute_metrics(poses[:, :3, :3], poses[:, :3, 3])
    return metrics["trans_path"] + np.deg2rad(metrics["rot_path_deg"])


def camera_in_part_frame(calibration, reference_calibration, reference_part_poses):
    """Camera-to-world in the moving part's frame at the first timestamp."""
    relative = np.linalg.inv(reference_part_poses[0]) @ reference_part_poses
    return reference_calibration @ relative @ np.linalg.inv(reference_calibration) @ calibration


def prepare_multiview(scene, camera_datasets, output):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if output == scene or output in scene.parents:
        raise ValueError("Multi-view output must not replace the raw scene")
    calibrations = calibration_poses(scene)
    cameras = sorted(camera_datasets)
    if len(cameras) < 2:
        raise ValueError("Joint multi-view training requires at least two cameras")
    frames = {}
    pose_sequences = {}
    assignments = {}
    pose_validity = {}
    entries = {}
    source_stems = None
    for camera in cameras:
        data = Path(camera_datasets[camera])
        conversion = json.loads((data / "metadata/freeart_conversion.json").read_text())
        if Path(conversion["scene_root"]).resolve() != scene or conversion["selected_camera"] != camera:
            raise ValueError(f"Camera sequence belongs to another source: {data}")
        stems = [row["source_stem"] for row in conversion["frame_map"]]
        if source_stems is not None and stems != source_stems:
            raise ValueError(f"Unsynchronized frame stems for {camera}")
        source_stems = stems
        frames[camera] = conversion["frame_map"]
        entries[camera] = load_camera_entry(scene, camera)
        for row in frames[camera]:
            stem = row["output_stem"]
            required = [data / folder / f"{stem}.png"
                        for folder in ("images", "images_ori", "masks", "mask_0", "mask_1")]
            required.append(data / "depth" / f"{stem}.npz")
            for path in required:
                if not path.is_file():
                    raise FileNotFoundError(f"Incomplete tracked camera sequence: {path}")
        pose_sequences[camera] = [np.load(data / f"global_poses_{part}.npy") for part in (0, 1)]
        pose_validity[camera] = []
        for part, poses in enumerate(pose_sequences[camera]):
            if poses.shape != (len(stems), 4, 4) or not np.isfinite(poses).all():
                raise ValueError(f"Invalid tracked pose sequence: {data}, part {part}: "
                                 f"shape {poses.shape}; expected {(len(stems), 4, 4)} finite poses. "
                                 "Recompute ICP with RGB timestamp slots preserved.")
            valid_path = data / f"pose_valid_{part}.npy"
            valid = np.load(valid_path) if valid_path.is_file() else np.ones(len(stems), dtype=bool)
            transforms_path = data / f"transforms_{part}.json"
            if not valid_path.is_file() and transforms_path.is_file():
                exported = json.loads(transforms_path.read_text())
                valid_names = {Path(row["file_path"]).stem for row in exported["frames"]}
                valid = np.array([row["output_stem"] in valid_names for row in frames[camera]])
            if valid.shape != (len(stems),):
                raise ValueError(f"Invalid ICP validity mask: {valid_path}")
            pose_validity[camera].append(valid.astype(bool))
        scores = [motion_score(poses) for poses in pose_sequences[camera]]
        moving = int(np.argmax(scores))
        assignments[camera] = {"moving_part": moving, "motion_scores": scores,
                               "valid_pose_counts": [int(valid.sum()) for valid in pose_validity[camera]]}
        if camera not in calibrations:
            raise ValueError(f"Camera {camera} missing from COLMAP images.txt")
    reference = max(cameras, key=lambda name: (
        bool(pose_validity[name][assignments[name]["moving_part"]][0]),
        assignments[name]["valid_pose_counts"][assignments[name]["moving_part"]],
        abs(np.diff(assignments[name]["motion_scores"])[0])))
    # Independent two-part tracking can mislabel static surfaces when the moving part is occluded.
    reference_motion = pose_sequences[reference][assignments[reference]["moving_part"]]
    moving_poses = {camera: camera_in_part_frame(calibrations[camera], calibrations[reference], reference_motion)
                    for camera in cameras}
    manifest = {"source_scene": str(scene), "cameras": cameras, "reference_camera": reference,
                "initialization_method": "independent_camera_segmentation_single_view_motion",
                "part_consistency_validated": False,
                "num_timestamps": len(source_stems), "part_assignments": assignments,
                "calibration_images": str(scene / "colmap/sparse/0/images.txt"), "frames": []}
    if output.exists():
        provenance = output / "multiview.json"
        if not provenance.exists() or json.loads(provenance.read_text())["source_scene"] != str(scene):
            raise ValueError(f"Refusing to overwrite an unrelated dataset: {output}")
    for directory in ("images", "images_ori", "masks", "mask_0", "mask_1", "depth"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    # Keep ownership recorded if a copy is interrupted so the same job can be retried.
    (output / "multiview.json").write_text(json.dumps(manifest, indent=2))
    transforms = [{"frames": [], "camera_model": "OPENCV"} for _ in range(2)]
    for timestamp, stem in enumerate(source_stems):
        for camera in cameras:
            data = Path(camera_datasets[camera])
            local_stem = frames[camera][timestamp]["output_stem"]
            name = f"{len(manifest['frames']):05d}"
            entry = entries[camera]
            k = camera_intrinsics(entry)
            row = dict(file_path=f"images/{name}.png", time_index=timestamp, camera_name=camera,
                       source_stem=stem, fl_x=float(k[0, 0]), fl_y=float(k[1, 1]),
                       cx=float(k[0, 2]), cy=float(k[1, 2]), w=entry["width"], h=entry["height"],
                       calibration_source=entry["intrinsics_source"])
            manifest["frames"].append(row)
            for folder in ("images", "images_ori", "masks"):
                shutil.copy2(data / folder / f"{local_stem}.png", output / folder / f"{name}.png")
            shutil.copy2(data / "depth" / f"{local_stem}.npz", output / "depth" / f"{name}.npz")
            moving = assignments[camera]["moving_part"]
            for part in (0, 1):
                source_part = moving if part == 1 else 1 - moving
                shutil.copy2(data / f"mask_{source_part}" / f"{local_stem}.png",
                             output / f"mask_{part}" / f"{name}.png")
                pose = moving_poses[camera][timestamp] if part == 1 else calibrations[camera]
                transforms[part]["frames"].append(dict(row, transform_matrix=pose.tolist(),
                    mask_path=f"mask_{part}/{name}.png", depth_path=f"depth/{name}.npz"))
    for part in (0, 1):
        (output / f"transforms_{part}.json").write_text(json.dumps(transforms[part], indent=2))
    np.savetxt(output / "cam_K.txt", camera_intrinsics(entries[cameras[0]]))
    (output / "multiview.json").write_text(json.dumps(manifest, indent=2))
    print(f"Combined {len(cameras)} cameras × {len(source_stems)} timestamps into {output}", flush=True)
    return manifest
