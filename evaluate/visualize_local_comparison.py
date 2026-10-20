"""Visualize estimated joint axes and saved renders, with optional joint GT."""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate.evaluate_local_joint_axis import align_prediction, evaluate_joint_axis
from preprocessing.prepare_multiview import calibration_poses


def raw_reference_image(scene, view, masked=True):
    stem, camera = view["source_stem"], view["camera_name"]
    rgb = np.asarray(Image.open(scene / "rgb" / camera / f"{stem}.png").convert("RGB"))
    if not masked:
        return Image.fromarray(rgb)
    mask = np.asarray(Image.open(scene / "mask" / camera / f"{stem}.png").convert("L")) > 127
    if rgb.shape[:2] != mask.shape:
        raise ValueError("Raw GT RGB and foreground mask dimensions differ")
    return Image.fromarray(np.where(mask[..., None], rgb, 255).astype(np.uint8))


def comparison_annotations(scene, dataset, results, manifest):
    articulation = json.loads((results / "articulation.json").read_text())
    if (scene / "pointcloud/joint_metadata.json").is_file():
        report = evaluate_joint_axis(scene, dataset, results)
        report["gt_available"] = True
    else:
        cameras = json.loads((results / "transforms_train.json").read_text())
        report = align_prediction(scene, manifest, articulation, cameras)
        report["gt_available"] = False
    if "predicted_origin" not in report:
        raise ValueError("Exported joint origin is required to project the estimated axis")
    joint_type = articulation["joint_type"]
    if joint_type not in ("prismatic", "revolute"):
        raise ValueError(f"Unsupported joint type: {joint_type}")
    key = "distance" if joint_type == "prismatic" else "angle_rad"
    predicted = {row["frame_index"]: float(row[key]) for row in articulation["frames"]}
    times = {row["time_index"] for row in manifest["frames"]}
    if (not times or set(predicted) != times or len(predicted) != len(articulation["frames"])
            or not np.isfinite(list(predicted.values())).all()):
        raise ValueError("Joint states must cover each timestamp exactly once and be finite")
    source_stems = {}
    for row in manifest["frames"]:
        t, stem = row["time_index"], row["source_stem"]
        if source_stems.setdefault(t, stem) != stem:
            raise ValueError("Camera views at one timestamp have different source frames")
    states = {t: dict(source_frame_index=int(source_stems[t].rsplit("_", 1)[-1]),
                      predicted_relative_state=predicted[t] - predicted[min(times)]) for t in times}
    if report.get("joint_state_evaluation") == "relative_to_first_timestamp":
        states = {row["time_index"]: row for row in report["joint_state_samples"]}
    report.update(predicted_joint_type=joint_type,
                  joint_state_unit="m" if joint_type == "prismatic" else "rad")
    return report, states


def project_axis(axis, origin, c2w, intrinsics, length):
    points = np.asarray(origin)[None, :] + np.array([[-.5], [.5]]) * length * np.asarray(axis)
    w2c = np.linalg.inv(c2w)
    points = points @ w2c[:3, :3].T + w2c[:3, 3]
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 1e-4):
        return None
    pixels = points @ intrinsics.T
    return pixels[:, :2] / pixels[:, 2:]


def group_views(manifest):
    groups = {}
    cameras = sorted(manifest["cameras"])
    for view in manifest["frames"]:
        group = groups.setdefault(view["time_index"], {})
        if view["camera_name"] in group:
            raise ValueError("Duplicate camera observation at one timestamp")
        group[view["camera_name"]] = view
    if not groups or set(groups) != set(range(len(groups))):
        raise ValueError("Timestamps must be contiguous starting at zero")
    if any(sorted(group) != cameras for group in groups.values()):
        raise ValueError("Each timestamp must contain all cameras")
    return cameras, groups


def draw_axis(image, pixels, color):
    if pixels is None:
        return
    draw = ImageDraw.Draw(image)
    a, b = pixels
    direction = b - a
    size = np.linalg.norm(direction)
    if size < 3:
        x, y = (a + b) / 2
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
        return
    direction /= size
    normal = np.array([-direction[1], direction[0]])
    # A white outline keeps the estimated red axis visible on red objects.
    draw.line([tuple(a), tuple(b)], fill="white", width=7)
    draw.line([tuple(a), tuple(b)], fill=color, width=3)
    draw.polygon([tuple(b), tuple(b - 8 * direction + 4 * normal),
                  tuple(b - 8 * direction - 4 * normal)], fill=color)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: <results>/comparison.")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("fps must be positive")
    scene, dataset, results = [p.resolve() for p in (args.scene, args.dataset, args.results)]
    manifest = json.loads((dataset / "multiview.json").read_text())
    cameras, groups = group_views(manifest)
    # Validate every saved render before writing comparison outputs.
    for group in groups.values():
        for view in group.values():
            path = results / "eval_results" / f"rgb_{Path(view['file_path']).stem}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing saved checkpoint render: {path}")
    report, states = comparison_annotations(scene, dataset, results, manifest)
    poses = calibration_poses(scene)
    out = args.output_dir.resolve() if args.output_dir else results / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    columns = min(2, len(cameras))
    width, height = columns * 640, 64 + math.ceil(len(cameras) / columns) * 280
    is_prismatic = report["predicted_joint_type"] == "prismatic"
    has_gt = report["gt_available"]
    has_gt_origin = "gt_origin" in report
    # Prismatic pivot location is unidentifiable; use the same display anchor for both directions.
    reference_anchor = report.get("gt_origin", report["predicted_origin"])
    anchors = [reference_anchor, reference_anchor if is_prismatic else report["predicted_origin"]]
    axes = [report.get("gt_axis", report["predicted_axis"]), report["predicted_axis"]]
    colors = ["#008833" if has_gt else "#dd2222", "#dd2222"]
    unit = report["joint_state_unit"]
    scale, display_unit = (180 / math.pi, "deg") if unit == "rad" else (1, "m")
    basename = "motion_vs_gt" if has_gt else "motion_axis_debug"
    with tempfile.TemporaryDirectory(prefix="freeartgs-comparison-") as temp:
        for t, group in sorted(groups.items()):
            state = states[t]
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            state_label = f"Estimated {state['predicted_relative_state'] * scale:.2f} {display_unit}"
            if "gt_relative_state" in state:
                state_label += f" | GT {state['gt_relative_state'] * scale:.2f} {display_unit}"
            draw.text((10, 5), f"{scene.name} | Frame {state['source_frame_index']} | {state_label}",
                font=font, fill="black")
            draw.text((10, 28), "Saved training views; axis overlays ignore occlusion. " + (
                "Prismatic axis anchor is arbitrary." if is_prismatic else
                "GT axis green; estimated axis red." if has_gt else "No joint GT; estimated axis red in both images."),
                font=font, fill="black")
            for c, camera in enumerate(cameras):
                view = group[camera]
                stem = Path(view["file_path"]).stem
                images = [raw_reference_image(scene, view, masked=has_gt),
                          Image.open(results / f"eval_results/rgb_{stem}.png").convert("RGB")]
                k = np.array([[view["fl_x"], 0, view["cx"]],
                              [0, view["fl_y"], view["cy"]], [0, 0, 1]])
                x, y = (c % columns) * 640, 64 + (c // columns) * 280
                for j, image in enumerate(images):
                    if image.size != (view["w"], view["h"]):
                        raise ValueError(f"Image dimensions differ from calibration: {stem}")
                    image = image.resize((320, 240), Image.Resampling.LANCZOS)
                    pixels = project_axis(axes[j], anchors[j], poses[camera], k, .4)
                    if pixels is not None:
                        pixels *= np.array([320 / view["w"], 240 / view["h"]])
                    draw_axis(image, pixels, colors[j])
                    draw.text((x + j * 320 + 8, y), f"{camera} | " + (
                        ("Reference / GT axis" if has_gt else "Recorded / estimated axis")
                        if j == 0 else "Render / estimated axis"), font=font, fill=colors[j])
                    canvas.paste(image, (x + j * 320, y + 24))
            canvas.save(Path(temp) / f"{t:05d}.png")
            if t in (0, len(groups) // 2, len(groups) - 1):
                canvas.save(out / f"comparison_{t:05d}.png")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(args.fps),
            "-i", str(Path(temp) / "%05d.png"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(out / f"{basename}.mp4")], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(args.fps),
            "-i", str(Path(temp) / "%05d.png"), "-filter_complex",
            "scale=960:-1,split[a][b];[a]palettegen[p];[b][p]paletteuse", "-loop", "0",
            str(out / f"{basename}.gif")], check=True)
    (out / "comparison_info.json").write_text(json.dumps(dict(report,
        source="Saved checkpoint renders versus recorded RGB" + (" composited on white using object masks" if has_gt else ""),
        cameras=cameras, timestamps=len(groups), fps=args.fps,
        views="training views; not novel-view evaluation", prismatic_common_display_anchor=is_prismatic,
        gt_origin_available=has_gt_origin, video=f"{basename}.mp4", animation=f"{basename}.gif"), indent=2))
    print(f"Motion comparison: {out / (basename + '.mp4')}")
    print(f"Animated preview: {out / (basename + '.gif')}")


if __name__ == "__main__":
    main()
