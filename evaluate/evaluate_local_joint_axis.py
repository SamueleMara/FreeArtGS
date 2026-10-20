"""Evaluate exported multi-view joint axes against render-world ground truth."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.prepare_multiview import calibration_poses


def unit_axis(value):
    axis = np.asarray(value, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError("Joint axis must be a finite nonzero 3-vector")
    return axis / np.linalg.norm(axis)


def load_gt_axis(scene):
    path = Path(scene) / "pointcloud/joint_metadata.json"
    gt = json.loads(path.read_text())
    if gt.get("coordinate_frame") != "render_world":
        raise ValueError(f"Expected render_world ground truth in {path}")
    return unit_axis(gt["axis_vector"])


def joint_motion_metrics(scene, manifest, articulation, axis_sign):
    """Compare relative states at matching source timestamps; never fit GT scale."""
    path = scene / "metadata/frame_values.csv"
    if not path.exists() or "frames" not in articulation:
        return dict(joint_state_evaluation="unavailable")
    gt = json.loads((scene / "pointcloud/joint_metadata.json").read_text())
    gt_type = "revolute" if gt.get("type") == "continuous" else gt.get("type")
    pred_type = articulation.get("joint_type")
    if gt_type not in ("revolute", "prismatic") or pred_type not in ("revolute", "prismatic"):
        raise ValueError("Unsupported or missing joint type for state evaluation")
    metrics = dict(gt_joint_type=gt_type, predicted_joint_type=pred_type,
                   joint_type_correct=gt_type == pred_type)
    if gt_type != pred_type:
        return dict(metrics, joint_state_evaluation="type_mismatch")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "frame_index" not in reader.fieldnames or gt["name"] not in reader.fieldnames:
            raise ValueError(f"GT trajectory columns do not match joint metadata: {path}")
        values = {int(row["frame_index"]): float(row[gt["name"]]) for row in reader}
    sources = {}
    for row in manifest["frames"]:
        source = int(row["source_stem"].rsplit("_", 1)[-1])
        if sources.setdefault(row["time_index"], source) != source:
            raise ValueError("Views at one timestamp map to different GT frames")
    key = "distance" if pred_type == "prismatic" else "angle_rad"
    predicted = {row["frame_index"]: float(row[key]) for row in articulation["frames"]}
    if len(predicted) != len(articulation["frames"]) or set(predicted) != set(sources):
        raise ValueError("Joint states must cover each shared timestamp exactly once")
    times = sorted(sources)
    if not times or any(sources[t] not in values for t in times):
        raise ValueError("GT trajectory does not cover the selected source timestamps")
    pred = np.array([predicted[t] for t in times]) * axis_sign
    target = np.array([values[sources[t]] for t in times])
    if not np.isfinite([pred, target]).all():
        raise ValueError("Joint states must be finite")
    # Canonical geometry and q=0 are arbitrary; only the first-state offset is removed.
    pred -= pred[0]
    target -= target[0]
    errors = np.abs(pred - target)
    metrics.update(joint_state_evaluation="relative_to_first_timestamp",
                   joint_state_unit="m" if pred_type == "prismatic" else "rad",
                   joint_state_mae=float(errors.mean()), joint_state_max_error=float(errors.max()),
                   predicted_motion_range=float(np.ptp(pred)), gt_motion_range=float(np.ptp(target)),
                   evaluated_timestamps=len(times), gt_trajectory_file=str(path))
    metrics["joint_state_samples"] = [dict(time_index=t, source_frame_index=sources[t],
        predicted_relative_state=float(p), gt_relative_state=float(g), absolute_error=float(e))
        for t, p, g, e in zip(times, pred, target, errors)]
    return metrics


def align_prediction(scene, manifest, articulation, cameras):
    """Map an exported prediction into the calibrated scene frame without GT."""
    scene = Path(scene).resolve()
    if Path(manifest["source_scene"]).resolve() != scene:
        raise ValueError("Multi-view dataset and source scene do not match")
    views = {Path(frame["file_path"]).name: frame for frame in manifest["frames"]}
    if not cameras:
        raise ValueError("No exported cameras available to align the predicted joint axis")
    if len({Path(row["file_path"]).name for row in cameras}) != len(cameras):
        raise ValueError("Exported camera observations must have unique filenames")
    calibrated = calibration_poses(scene)
    alignment = None
    translation = None
    camera_errors = []
    for camera in cameras:
        view = views[Path(camera["file_path"]).name]
        rotation = np.asarray(camera["transform"], dtype=np.float64)[:3, :3]
        if not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError("Exported camera rotation is invalid")
        world_rotation = calibrated[view["camera_name"]][:3, :3]
        if alignment is None:
            # The blend parser anchors its world to the first camera, rather than render_world.
            alignment = world_rotation @ rotation.T
            translation = calibrated[view["camera_name"]][:3, 3] - alignment @ np.asarray(camera["transform"])[:3, 3]
        if not np.allclose(alignment @ rotation, world_rotation, atol=1e-4):
            raise ValueError("Exported camera rotations are inconsistent with the fixed COLMAP calibration")
        camera_errors.append(float(np.linalg.norm(
            alignment @ np.asarray(camera["transform"])[:3, 3] + translation
            - calibrated[view["camera_name"]][:3, 3])))
    predicted_axis = unit_axis(alignment @ unit_axis(articulation["joint_axis_world"]))
    report = dict(predicted_axis=predicted_axis.tolist(),
                  prediction_to_render_rotation=alignment.tolist(),
                  prediction_to_render_translation=translation.tolist(),
                  camera_position_max_error_m=max(camera_errors),
                  exported_observations=len(cameras), expected_observations=len(views))
    if "joint_origin_world" in articulation:
        origin = np.asarray(articulation["joint_origin_world"], dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Joint origin must be a finite 3-vector")
        report["predicted_origin"] = (alignment @ origin + translation).tolist()
    return report


def evaluate_joint_axis(scene, dataset, results):
    scene, dataset, results = map(lambda p: Path(p).resolve(), (scene, dataset, results))
    gt_axis = load_gt_axis(scene)
    manifest = json.loads((dataset / "multiview.json").read_text())
    articulation = json.loads((results / "articulation.json").read_text())
    cameras = json.loads((results / "transforms_train.json").read_text())
    report = align_prediction(scene, manifest, articulation, cameras)
    predicted_axis = np.asarray(report["predicted_axis"])
    alignment = np.asarray(report["prediction_to_render_rotation"])
    translation = np.asarray(report["prediction_to_render_translation"])
    cosine = np.clip(abs(np.dot(predicted_axis, gt_axis)), 0.0, 1.0)
    report.update(object=scene.name, joint_axis_error_deg=float(np.degrees(np.arccos(cosine))),
                  metric="acos(abs(dot(unit(pred_axis), unit(gt_axis)))) in degrees",
                  coordinate_frame="render_world", predicted_axis=predicted_axis.tolist(),
                  gt_axis=gt_axis.tolist(),
                  gt_file=str(scene / "pointcloud/joint_metadata.json"),
                  prediction_file=str(results / "articulation.json"),
                  evaluation_scope="local calibrated-data diagnostics; not the FreeArt-21 benchmark",
                  reconstruction_quality_evaluated=False)
    report.update(joint_motion_metrics(scene, manifest, articulation,
                                      1.0 if np.dot(predicted_axis, gt_axis) >= 0 else -1.0))
    gt_metadata = json.loads((scene / "pointcloud/joint_metadata.json").read_text())
    if "joint_origin_world" in articulation and "point_of_application" in gt_metadata:
        gt_origin = np.asarray(gt_metadata["point_of_application"], dtype=np.float64)
        pred_origin = np.asarray(articulation["joint_origin_world"], dtype=np.float64)
        if gt_origin.shape != (3,) or pred_origin.shape != (3,) or not np.isfinite([gt_origin, pred_origin]).all():
            raise ValueError("Joint origins must be finite 3-vectors")
        report.update(gt_origin=gt_origin.tolist(), predicted_origin=(alignment @ pred_origin + translation).tolist(),
                      gt_axis_in_prediction_frame=unit_axis(alignment.T @ gt_axis).tolist(),
                      gt_origin_in_prediction_frame=(alignment.T @ (gt_origin - translation)).tolist())
        if gt_metadata.get("type") == "prismatic":
            report.update(joint_axis_position_error_m=None,
                          joint_axis_position_evaluation="undefined for a prismatic joint")
        elif gt_metadata.get("type") in ("revolute", "continuous"):
            delta = np.asarray(report["predicted_origin"]) - gt_origin
            normal = np.cross(predicted_axis, gt_axis)
            distance = (abs(np.dot(delta, normal)) / np.linalg.norm(normal)
                        if np.linalg.norm(normal) > 1e-6
                        else np.linalg.norm(np.cross(delta, gt_axis)))
            report["joint_axis_position_error_m"] = float(distance)
    (results / "joint_axis_error.json").write_text(json.dumps(report, indent=2))
    if "gt_origin" in report and (results / "object_3dgs.ply").is_file():
        save_axis_preview(results, report)
    return report


def save_axis_preview(results, report):
    """Show exported Gaussian centres and the two axis lines in render_world."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plyfile import PlyData

    vertices = PlyData.read(str(results / "object_3dgs.ply"))["vertex"].data
    indices = np.linspace(0, len(vertices) - 1, min(10000, len(vertices)), dtype=int)
    points = np.column_stack([vertices[name][indices] for name in ("x", "y", "z")])
    rotation = np.asarray(report["prediction_to_render_rotation"])
    points = points @ rotation.T + np.asarray(report["prediction_to_render_translation"])
    colors = (np.column_stack([vertices[name][indices] for name in ("red", "green", "blue")]) / 255
              if all(name in vertices.dtype.names for name in ("red", "green", "blue")) else "gray")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*points.T, c=colors, s=1, alpha=.5, label="Gaussian centres")
    center = (points.min(0) + points.max(0)) / 2
    length = max(float(np.linalg.norm(np.ptp(points, axis=0))), .01)
    endpoints = [points]
    for prefix, color, label in (("predicted", "red", "Predicted axis"), ("gt", "green", "GT axis")):
        axis, origin = np.asarray(report[f"{prefix}_axis"]), np.asarray(report[f"{prefix}_origin"])
        anchor = origin + axis * np.dot(center - origin, axis)
        line = anchor + np.array([[-.6], [.6]]) * length * axis
        ax.plot(*line.T, color=color, linewidth=2, label=label)
        endpoints.append(line)
    bounds = np.concatenate(endpoints)
    center = (bounds.min(0) + bounds.max(0)) / 2
    radius = max(float(np.ptp(bounds, axis=0).max()) / 2, .01)
    for setter, coordinate in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(coordinate - radius, coordinate + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)",
           title=f"{report['object']} — axis direction error {report['joint_axis_error_deg']:.6f}°")
    if report.get("gt_joint_type") == "prismatic":
        ax.set_title(f"{report['object']} — axis direction error {report['joint_axis_error_deg']:.6f}°\n"
                     "Prismatic joint: axis origins do not affect motion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "joint_axis_preview.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="Raw scene containing ground-truth metadata.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared shared multi-view dataset.")
    parser.add_argument("--results", type=Path, required=True, help="splatfacto-art output directory.")
    args = parser.parse_args()
    report = evaluate_joint_axis(args.scene, args.dataset, args.results)
    print(f"{report['object']}: joint axis error = {report['joint_axis_error_deg']:.6f} degrees")
    print(f"Report: {args.results / 'joint_axis_error.json'}")


if __name__ == "__main__":
    main()
