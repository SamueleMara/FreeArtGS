#!/usr/bin/env python3
"""Convert a calibrated RGB-D recording to the standard multiview layout, without masks.

Requires numpy, opencv-python and scipy. Sources are read only. Output must be new.
RGB is undistorted to one common centered PINHOLE K. Raw depth is unprojected,
transformed by T_color_depth and z-buffered into that rectified RGB camera.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from point_sampling import point_fraction, sample_pixels


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def transform(value, label):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label}: expected a finite 4x4 transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5):
        raise ValueError(f"{label}: invalid rigid transform")
    return matrix


def intrinsics(camera):
    matrix = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera["D"], dtype=np.float64)
    if camera["distortion_model"] not in ("plumb_bob", "rational_polynomial"):
        raise ValueError(f"Unsupported distortion model: {camera['distortion_model']}")
    if distortion.shape not in ((4,), (5,), (8,), (12,), (14,)) or not np.isfinite(distortion).all():
        raise ValueError("Unsupported or invalid distortion coefficients")
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("Invalid pinhole intrinsics")
    return matrix, distortion


def inspect_recording(source, episode, size, limit):
    calibration = json.loads((source / "meta/camera_calibration.json").read_text())
    info = json.loads((source / "meta/info.json").read_text())
    if calibration["extrinsic_convention"] != "T_base_color_optical; p_base = T_base_color_optical * p_color_optical":
        raise ValueError("Unsupported camera extrinsic convention; expected optical camera-to-base transforms")
    cameras, common = {}, None
    for name, camera in sorted(calibration["cameras"].items()):
        if not re.fullmatch(r"cam_\d+", name):
            raise ValueError(f"Unsupported camera name: {name}")
        intrinsics(camera["color"]); intrinsics(camera["depth"])
        if camera["depth"]["units"] != "uint16_mm":
            raise ValueError(f"{name}: unsupported raw depth units")
        transform(camera["T_color_depth"], name + "/T_color_depth")
        transform(camera["T_base_color_optical"], name + "/T_base_color_optical")
        video = source / f"videos/chunk-000/observation.images.{name}/episode_{episode:06d}.mp4"
        depth_dir = source / f"depth/chunk-000/observation.depth.{name}/episode_{episode:06d}"
        depth_files = {int(p.stem.removeprefix("frame_")): p for p in depth_dir.glob("frame_*.png")}
        cap = cv2.VideoCapture(str(video))
        try:
            count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
            ok, image = cap.read()
            if not ok or image.shape[:2] != (camera["color"]["height"], camera["color"]["width"]):
                raise ValueError(f"{name}: missing RGB video or size differs from calibration")
            if not np.isclose(fps, info["fps"], atol=.01):
                raise ValueError(f"{name}: video FPS differs from recording metadata")
        finally:
            cap.release()
        if not depth_files or any(t < 0 or t >= count for t in depth_files):
            raise ValueError(f"{name}: depth frame indices are missing or outside the RGB video")
        available = set(depth_files)
        common = available if common is None else common & available
        cameras[name] = dict(calibration=camera, video=video, depth_files=depth_files, rgb_frames=count)
    if not cameras or not common:
        raise ValueError("No synchronized RGB/depth frames across calibrated cameras")
    frames = sorted(common)
    if limit:
        frames = frames[:limit]
    width, height = size
    fx = np.mean([c["calibration"]["color"]["K"][0] * width / c["calibration"]["color"]["width"] for c in cameras.values()])
    fy = np.mean([c["calibration"]["color"]["K"][4] * height / c["calibration"]["color"]["height"] for c in cameras.values()])
    target_k = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], dtype=np.float64)
    return calibration, info, cameras, frames, target_k


def depth_rays(camera):
    matrix, distortion = intrinsics(camera)
    yy, xx = np.mgrid[:camera["height"], :camera["width"]]
    pixels = np.stack([xx, yy], axis=-1).astype(np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPointsIter(pixels, matrix, distortion, None, None,
                                 (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)).reshape(-1, 2)
    rays = np.column_stack([rays, np.ones(len(rays))])
    # Reject inverse-distortion failures rather than producing displaced geometry.
    roundtrip, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), matrix, distortion)
    valid = np.isfinite(rays).all(axis=1) & (np.linalg.norm(roundtrip.reshape(-1, 2) - pixels.reshape(-1, 2), axis=1) < .25)
    return rays, valid


def register_depth(raw, rays, ray_valid, color_from_depth, target_k, size):
    if raw.dtype != np.uint16 or raw.size != len(rays):
        raise ValueError("Raw depth must be uint16 with calibrated dimensions")
    width, height = size
    z = raw.ravel().astype(np.float64) * .001
    accepted = ray_valid & (z > 0)
    points = (rays[accepted] * z[accepted, None]) @ color_from_depth[:3, :3].T + color_from_depth[:3, 3]
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & (points[:, 2] <= 65.535)]
    projection = points @ target_k.T
    pixels = np.rint(projection[:, :2] / projection[:, 2:]).astype(np.int64)
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    pixels, points = pixels[inside], points[inside]
    depth = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(depth, pixels[:, 1] * width + pixels[:, 0], points[:, 2].astype(np.float32))
    depth[~np.isfinite(depth)] = 0
    return depth.reshape(height, width)


def selected_rgb(video, frames):
    """Decode selected source indices; the caller assigns consecutive output indices."""
    cap = cv2.VideoCapture(str(video))
    cursor = -1
    try:
        for index in frames:
            while cursor < index:
                if not cap.grab():
                    raise ValueError(f"Video ended before frame {index}: {video}")
                cursor += 1
            ok, image = cap.retrieve()
            if not ok:
                raise ValueError(f"Could not decode frame {index}: {video}")
            yield index, image
    finally:
        cap.release()


def save_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write {path}")


def write_ply(path, xyz, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for position, color in zip(xyz, rgb):
            stream.write(" ".join(map(str, [*position, 0, 0, 0, *color])) + "\n")


def merge_endpoint_clouds(output, cameras, images, size, target_k, stride, voxel_size, state, fraction=None):
    """Fuse only cameras at one time; retain one depth observation per view/voxel."""
    xyzs, colors, samples = [], [], []
    camera_by_id = {c["id"]: c for c in cameras}
    for image in images:
        camera = camera_by_id[image["camera_id"]]
        depth = np.load(output / "depth_npy" / Path(image["name"]).with_suffix(".npy"), allow_pickle=False)
        rgb = cv2.imread(str(output / "rgb" / image["name"]))[:, :, ::-1]
        step = stride if fraction is None else 1
        yy, xx = np.mgrid[0:size[1]:step, 0:size[0]:step]
        z = depth[yy, xx]
        valid = np.isfinite(z) & (z > 0)
        x, y, z = xx[valid], yy[valid], z[valid]
        if fraction is not None:
            y, x = sample_pixels(y, x, fraction)
            z = depth[y, x]
        local = np.column_stack([(x - target_k[0, 2]) / target_k[0, 0] * z,
                                 (y - target_k[1, 2]) / target_k[1, 1] * z, z])
        pose = np.asarray(camera["pose_matrix_4x4"])
        world = local @ pose[:3, :3].T + pose[:3, 3]
        color = rgb[y, x]
        write_ply(output / f"pointcloud/views/{state}/{camera['name']}.ply", world, color)
        xyzs.append(world); colors.append(color)
        samples.extend((image, int(u), int(v)) for u, v in zip(x, y))
    xyz, rgb = np.concatenate(xyzs), np.concatenate(colors)
    if len(xyz) == 0:
        raise ValueError(f"{state}: no valid sampled depth points; reduce --point-stride")
    _, inverse = np.unique(np.floor(xyz / voxel_size).astype(np.int64), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    fused = np.column_stack([np.bincount(inverse, weights=xyz[:, axis]) / count for axis in range(3)])
    fused_rgb = np.rint(np.column_stack([np.bincount(inverse, weights=rgb[:, axis]) / count for axis in range(3)])).clip(0, 255).astype(np.uint8)
    selected = [{} for _ in count]
    for index, group in enumerate(inverse):
        image, _, _ = samples[index]
        previous = selected[group].get(image["id"])
        if previous is None or np.linalg.norm(xyz[index] - fused[group]) < np.linalg.norm(xyz[previous] - fused[group]):
            selected[group][image["id"]] = index
    points = []
    for group, members in enumerate(selected):
        pid, tracks, errors = group + 1, [], []
        for index in members.values():
            image, u, v = samples[index]
            tracks.append((image["id"], len(image["observations"])))
            image["observations"].append((u, v, pid))
            camera = camera_by_id[image["camera_id"]]
            pose = np.asarray(camera["pose_matrix_4x4"])
            local = pose[:3, :3].T @ (fused[group] - pose[:3, 3])
            projected = target_k @ local
            errors.append(float(np.sum((projected[:2] / projected[2] - [u, v]) ** 2)))
        points.append((pid, fused[group], fused_rgb[group].tolist(), float(np.sqrt(np.mean(errors))), tracks))
    return points, dict(source_points=len(xyz), merged_points=len(points), voxel_size_m=voxel_size,
                        point_fraction=fraction, point_stride=stride if fraction is None else 1,
                        multi_camera_points=sum(len(p[4]) > 1 for p in points), state=state)


def write_colmap(output, cameras, images, points, size, target_k, model_name="colmap", ply_name="points3D.ply"):
    expected_views = {camera["name"] for camera in cameras}
    actual_views = [str(Path(image["name"]).parent) for image in images]
    times = {Path(image["name"]).stem for image in images}
    if len(images) != len(expected_views) or set(actual_views) != expected_views or len(times) != 1:
        raise ValueError("Each COLMAP endpoint must contain exactly one image per view at one time")
    if model_name == "colmap" and times != {"frame_000000"}:
        raise ValueError("Initial COLMAP endpoint must use frame_000000")
    sparse = output / model_name / "sparse_unmasked/0"
    sparse.mkdir(parents=True)
    width, height = size
    params = [target_k[0, 0], target_k[1, 1], target_k[0, 2], target_k[1, 2]]
    with (sparse / "cameras.txt").open("w") as text, (sparse / "cameras.bin").open("wb") as binary:
        text.write("# CAMERA_ID MODEL WIDTH HEIGHT FX FY CX CY\n")
        binary.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            cid = camera["id"]
            text.write(f"{cid} PINHOLE {width} {height} " + " ".join(map(str, params)) + "\n")
            binary.write(struct.pack("<iiQQdddd", cid, 1, width, height, *params))
    with (sparse / "images.txt").open("w") as text, (sparse / "images.bin").open("wb") as binary:
        text.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME; next line: X Y POINT3D_ID\n")
        binary.write(struct.pack("<Q", len(images)))
        for item in images:
            text.write(" ".join(map(str, [item["id"], *item["q"], *item["t"], item["camera_id"], item["name"]])) + "\n")
            text.write(" ".join(str(v) for observation in item["observations"] for v in observation) + "\n")
            binary.write(struct.pack("<idddddddi", item["id"], *item["q"], *item["t"], item["camera_id"]))
            binary.write(item["name"].encode() + b"\0" + struct.pack("<Q", len(item["observations"])))
            for x, y, pid in item["observations"]:
                binary.write(struct.pack("<ddq", x, y, pid))
    with (sparse / "points3D.txt").open("w") as text, (sparse / "points3D.bin").open("wb") as binary:
        text.write("# Voxel-merged multiview depth at one endpoint; unmasked, not triangulated or part-labelled.\n")
        binary.write(struct.pack("<Q", len(points)))
        for pid, xyz, rgb, error, tracks in points:
            text.write(" ".join(map(str, [pid, *xyz, *rgb, error, *(v for track in tracks for v in track)])) + "\n")
            binary.write(struct.pack("<QdddBBBdQ", pid, *xyz, *rgb, error, len(tracks)))
            for image_id, point_index in tracks:
                binary.write(struct.pack("<ii", image_id, point_index))
    write_ply(output / "pointcloud" / ply_name, [p[1] for p in points], [p[2] for p in points])


def export_final_colmap(output, cameras, frame, size, target_k, stride, voxel_size, calibration_status, source_frame=None, fraction=None):
    if (output / "colmap_final").exists():
        raise FileExistsError(f"Final COLMAP model already exists: {output / 'colmap_final'}")
    images = []
    for camera in cameras:
        pose = np.linalg.inv(np.asarray(camera["pose_matrix_4x4"]))
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        images.append(dict(id=len(images) + 1, camera_id=camera["id"], name=f"{camera['name']}/frame_{frame:06d}.png",
                           q=quaternion, t=pose[:3, 3], observations=[]))
    points, fusion = merge_endpoint_clouds(output, cameras, images, size, target_k, stride, voxel_size, "end", fraction)
    write_colmap(output, cameras, images, points, size, target_k, "colmap_final", "points3D_final.ply")
    for item in images:
        destination = output / "colmap_final/images" / item["name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "rgb" / item["name"], destination)
    summary = dict(colmap_dir=str(output / "colmap_final"), sparse_dir=None, sparse_unmasked_dir=str(output / "colmap_final/sparse_unmasked/0"), masking_pending=True,
                   pointcloud_ply=str(output / "pointcloud/points3D_final.ply"), endpoint_frame=frame,
                   source_frame=frame if source_frame is None else source_frame,
                   num_cameras=len(cameras), num_images=len(images), num_points3D=len(points), fusion=fusion,
                   calibration_status=calibration_status, part_labels_available=False)
    write_json(output / "metadata/colmap_export_final.json", summary)
    write_json(output / "colmap_final/export_summary.json", summary)
    return summary


def add_final_to_existing(args):
    output = args.output.expanduser().resolve()
    if (output / "conversion_in_progress.json").exists():
        raise ValueError("Original RGB-D conversion is incomplete")
    metadata = json.loads((output / "metadata/real_conversion.json").read_text())
    cameras = json.loads((output / "metadata/cameras.json").read_text())["cameras"]
    size = (cameras[0]["width"], cameras[0]["height"])
    frame = metadata.get("output_frames", metadata["source_frames"])[-1]
    if args.dry_run:
        print(f"Would fuse {len(cameras)} RGB-D clouds at frame {frame} into {output / 'colmap_final'}")
        return
    summary = export_final_colmap(output, cameras, frame, size, np.asarray(metadata["rectified_K"]),
                                  args.point_stride, args.voxel_size, metadata.get("calibration_status"), metadata["source_frames"][-1],
                                  args.point_fraction if args.point_fraction is not None else metadata.get('point_fraction'))
    metadata["final_colmap"] = summary
    if args.point_fraction is not None:
        metadata['point_fraction'] = args.point_fraction
    write_json(output / "metadata/real_conversion.json", metadata)
    print(f"Added colmap_final: frame {frame}, {summary['num_points3D']} fused points; RGB/depth/masks unchanged", flush=True)


def convert(args):
    source, output = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Input and output must be separate, non-nested directories")
    calibration, info, sources, frames, target_k = inspect_recording(source, args.episode, args.size, args.max_frames)
    print(f"{len(sources)} cameras; {len(frames)} synchronized frames per camera; indices {frames[0]}..{frames[-1]}; output {args.size[0]}x{args.size[1]}", flush=True)
    print(f"Calibration status: {calibration.get('status', 'unspecified')}; preserved without claiming validation", flush=True)
    print(f"Output: {output}; masks will not be created", flush=True)
    if args.dry_run:
        return
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new folder: {output}")
    output.mkdir(parents=True)
    write_json(output / "conversion_in_progress.json", dict(source=str(source)))
    for folder in ["rgb", "depth_npy", "depth_png", "metadata/raw", "pointcloud", "colmap/images"]:
        (output / folder).mkdir(parents=True)
    cameras, images, statistics = [], [], {}
    for cid, (source_name, data) in enumerate(sources.items(), 1):
        camera = data["calibration"]
        name = f"cam_{int(source_name.removeprefix('cam_')):03d}"
        for folder in ["rgb", "depth_npy", "depth_png", "colmap/images"]:
            (output / folder / name).mkdir()
        matrix, distortion = intrinsics(camera["color"])
        maps = cv2.initUndistortRectifyMap(matrix, distortion, None, target_k, tuple(args.size), cv2.CV_32FC1)
        color_valid = (maps[0] >= 0) & (maps[0] < camera["color"]["width"] - 1) & (maps[1] >= 0) & (maps[1] < camera["color"]["height"] - 1)
        rays, ray_valid = depth_rays(camera["depth"])
        color_from_depth = transform(camera["T_color_depth"], "depth-to-color")
        world_from_color = transform(camera["T_base_color_optical"], "camera-to-world")
        world_to_color = np.linalg.inv(world_from_color)
        quaternion = Rotation.from_matrix(world_to_color[:3, :3]).as_quat()[[3, 0, 1, 2]]
        cameras.append(dict(id=cid, name=name, source_camera_name=source_name, width=args.size[0], height=args.size[1],
                            fx=float(target_k[0, 0]), fy=float(target_k[1, 1]), cx=float(target_k[0, 2]), cy=float(target_k[1, 2]),
                            pose_matrix_4x4=world_from_color.tolist(), pose_convention="opencv_optical_camera_to_world",
                            coordinate_frame=calibration["base_frame"]))
        coverage = []
        for number, (frame, rgb) in enumerate(selected_rgb(data["video"], frames)):
            raw = cv2.imread(str(data["depth_files"][frame]), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.shape != (camera["depth"]["height"], camera["depth"]["width"]):
                raise ValueError(f"{source_name}/{frame}: depth size differs from calibration")
            rgb = cv2.remap(rgb, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            depth = register_depth(raw, rays, ray_valid, color_from_depth, target_k, args.size)
            depth[~color_valid] = 0
            if not np.any(depth > 0):
                raise ValueError(f"{source_name}/{frame}: no depth projects into the rectified RGB image")
            coverage.append(float(np.mean(depth > 0)))
            filename = f"frame_{number:06d}"
            save_png(output / f"rgb/{name}/{filename}.png", rgb)
            if number == 0:
                shutil.copyfile(output / f"rgb/{name}/{filename}.png", output / f"colmap/images/{name}/{filename}.png")
            np.save(output / f"depth_npy/{name}/{filename}.npy", depth, allow_pickle=False)
            save_png(output / f"depth_png/{name}/{filename}.png", np.rint(depth * 1000).clip(0, 65535).astype(np.uint16))
            if number == 0:
                images.append(dict(id=len(images) + 1, camera_id=cid, name=f"{name}/{filename}.png",
                                   q=quaternion, t=world_to_color[:3, 3], observations=[]))
            if number % 50 == 0 or number == len(frames) - 1:
                print(f"{name}: {number + 1}/{len(frames)} frames", flush=True)
        statistics[name] = dict(frames=len(frames), mean_valid_depth_fraction=float(np.mean(coverage)),
                                rejected_depth_distortion_rays=int((~ray_valid).sum()))
    initial_images = images
    points, initial_fusion = merge_endpoint_clouds(output, cameras, initial_images, args.size, target_k, args.point_stride, args.voxel_size, "start", args.point_fraction)
    write_colmap(output, cameras, images, points, args.size, target_k)
    summary = dict(colmap_dir=str(output / "colmap"), sparse_dir=None, sparse_unmasked_dir=str(output / "colmap/sparse_unmasked/0"), masking_pending=True,
                   pointcloud_ply=str(output / "pointcloud/points3D.ply"), endpoint_frame=0, source_frame=frames[0],
                   num_cameras=len(cameras), num_images=len(images), num_points3D=len(points),
                   fusion=initial_fusion, calibration_status=calibration.get("status"), part_labels_available=False)
    write_json(output / "metadata/colmap_export.json", summary)
    write_json(output / "colmap/export_summary.json", summary)
    final_summary = export_final_colmap(output, cameras, len(frames) - 1, args.size, target_k, args.point_stride, args.voxel_size, calibration.get("status"), frames[-1], args.point_fraction)
    write_json(output / "metadata/cameras.json", dict(cameras=cameras, coordinate_frame=calibration["base_frame"], calibration_status=calibration.get("status")))
    for path in (source / "meta").iterdir():
        if path.is_file():
            shutil.copyfile(path, output / "metadata/raw" / path.name)
    parquet = source / f"data/chunk-000/episode_{args.episode:06d}.parquet"
    if parquet.exists():
        shutil.copyfile(parquet, output / "metadata/raw" / parquet.name)
    with (output / "metadata/frame_timestamps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["frame_index", "source_frame_index", "timestamp_s"])
        writer.writerows((number, frame, frame / float(info["fps"])) for number, frame in enumerate(frames))
    write_json(output / "metadata/real_conversion.json", dict(source=str(source), episode=args.episode, source_frames=frames,
               output_frames=list(range(len(frames))), frame_naming="consecutive_zero_based",
               frame_matching="recorded frame indices; hardware synchronization not independently validated",
               timestamps="source_frame_index / recording FPS; not hardware timestamps", depth_input_unit="uint16 millimeters",
               depth_output_unit="float32 meters, rectified color-camera Z; zero means missing", rectified_K=target_k.tolist(),
               coordinate_frame=calibration["base_frame"], calibration_status=calibration.get("status"), calibration_warning=calibration.get("warning"),
               point_fraction=args.point_fraction,
               masks_generated=False, joint_gt_available=False, seed_cloud="Unmasked multiview RGB-D fusion; first and last synchronized frames exported separately",
               endpoint_clouds=dict(start=initial_fusion, end=final_summary["fusion"]),
               cameras=statistics))
    (output / "README.txt").write_text("RGB, registered depth and COLMAP share one centered, undistorted PINHOLE K.\n"
        "depth_npy is camera-Z in meters; depth_png is in millimeters, zero for missing.\n"
        "Output frame IDs are consecutive from zero; original capture indices are preserved in metadata.\n"
        "Camera/world poses use original optical camera-to-base transforms; their original validation status is preserved.\n"
        "Masks and joint GT are absent. Generate whole-object masks in mask/cam_NNN/frame_NNNNNN.png using these rectified RGB images.\n"
        "Unmasked models are in sparse_unmasked/0; mask generation publishes masked models to sparse/0.\n"
        "colmap contains only frame_000000 from each camera and first-frame fused seeds; colmap_final uses last-selected-frame images and independently fused seeds.\n"
        "Per-camera endpoint clouds are in pointcloud/views/{start,end}; merged clouds are points3D.ply and points3D_final.ply.\n"
        "These clouds include background, robot and object and have no part labels. Temporal states are never fused together.\n")
    (output / "conversion_in_progress.json").unlink()
    print(f"Completed: {len(frames) * len(cameras)} RGB-D observations; {len(images)} views per endpoint, {len(points)} start / {final_summary['num_points3D']} end fused points; {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New named scene directory")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--point-fraction", type=point_fraction, help="Fraction (0, 1] of valid depth pixels per camera; overrides --point-stride, also saved for masked export")
    parser.add_argument("--voxel-size", type=float, default=.005, help="World-space voxel size in meters for multiview endpoint fusion")
    parser.add_argument("--max-frames", type=int, default=0, help="Smoke-test limit per camera; zero means all synchronized frames")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--renumber-frames", action="store_true", help="Repair an existing legacy conversion to consecutive frame names, including masks and COLMAP references")
    parser.add_argument("--final-colmap-only", action="store_true", help="Add colmap_final to an already converted output without re-decoding RGB/depth")
    args = parser.parse_args(argv)
    if args.episode < 0 or args.max_frames < 0 or min(*args.size, args.point_stride) < 1 or not np.isfinite(args.voxel_size) or args.voxel_size <= 0:
        parser.error("Invalid episode, frame limit, resolution or point stride")
    if args.renumber_frames and args.final_colmap_only:
        parser.error("Choose either --renumber-frames or --final-colmap-only")
    if args.renumber_frames:
        from renumber_real_recording import renumber
        renumber(args.output, dry_run=args.dry_run)
    elif args.final_colmap_only:
        add_final_to_existing(args)
    elif args.input is None:
        parser.error("--input is required for full conversion")
    else:
        convert(args)


if __name__ == "__main__":
    main()
