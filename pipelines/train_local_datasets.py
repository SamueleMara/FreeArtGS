#!/usr/bin/env python3
"""Train one shared FreeArtGS model per object using all calibrated cameras."""
import argparse
import errno
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import yaml
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render")
sys.path.insert(0, str(REPO_ROOT))

from samuele.convert_standard_to_freeart import (
    build_triplets, convert, load_camera_entry, maybe_write_config, parse_args as conversion_args,
)
from preprocessing.prepare_multiview import calibration_poses, prepare_multiview
from evaluate.evaluate_local_joint_axis import evaluate_joint_axis, load_gt_axis
from utils.local_storage import DEFAULT_OUTPUTS_ROOT, configure_runtime, ensure_free_space


def storage_failure(error, log_path):
    if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT):
        return True
    if isinstance(error, subprocess.CalledProcessError) and log_path.is_file():
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 65536))
            tail = log.read().lower()
        return any(message in tail for message in (b"no space left on device", b"disk quota exceeded",
                                                   b"no usable temporary directory"))
    return False


def write_summary(path, rows):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, indent=2))
    temporary.replace(path)


def result_is_complete(record, output, iterations, evaluate=False, visualize=False):
    if not record or record.get("status") != "passed" or not record.get("training_completed"):
        return False
    if visualize and record.get("visualization_error"):
        return False
    required = ["config.yml", f"nerfstudio_models/step-{iterations - 1:09d}.ckpt",
                "object_3dgs.ply", "3dgs_part_weight.txt", "articulation.json", "transforms_train.json"]
    json_files = ["articulation.json", "transforms_train.json"]
    if evaluate:
        required.append("joint_axis_error.json")
        json_files.append("joint_axis_error.json")
    if visualize:
        basename = "motion_vs_gt" if evaluate else "motion_axis_debug"
        required.extend([f"comparison/{basename}.mp4", f"comparison/{basename}.gif",
                         "comparison/comparison_info.json"])
        json_files.append("comparison/comparison_info.json")
    try:
        if any(not (output / name).is_file() or (output / name).stat().st_size == 0 for name in required):
            return False
        contents = {name: json.loads((output / name).read_text()) for name in json_files}
        if not contents["articulation.json"].get("frames") or not contents["transforms_train.json"]:
            return False
        if evaluate and not np.isfinite(contents["joint_axis_error.json"]["joint_axis_error_deg"]):
            return False
        if visualize and not contents["comparison/comparison_info.json"].get("timestamps"):
            return False
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
    return True


def discover_jobs(dataset_root):
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Raw datasets directory is unavailable: {dataset_root}. "
                                "Pass --datasets with the existing source directory.")
    scenes = sorted(p for p in dataset_root.iterdir() if p.is_dir()
                    and p.name != "episode_000000"
                    and (p / "metadata/cameras.json").is_file()
                    and any(d.is_dir() for d in (p / "rgb").glob("*")))
    if not scenes:
        raise ValueError(f"No calibrated multi-camera RGB-D datasets found in {dataset_root}")
    return scenes


def require_source(scene):
    if not scene.is_dir():
        raise FileNotFoundError(f"Raw source disappeared after validation: {scene}. "
                                "Do not move/delete source datasets during a run; retry with --datasets at their current location.")


def run_stage(config_path, log, stage):
    subprocess.run(["bash", "pipelines/freeartgs.sh", "--config", str(config_path), stage],
                   cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS_ROOT, help="Read-only raw source datasets.")
    parser.add_argument("--all-cameras", action="store_true", help="All cameras are used together by default.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--object", help="Optional raw object folder name to train.")
    selection.add_argument("--objects", nargs="+", help="Train exactly these raw object folder names, in order.")
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT,
                        help="Target for results, converted inputs, prepared datasets, configs, caches and temporary files.")
    parser.add_argument("--iterations", type=int, default=30000, help="Iterations per training stage.")
    parser.add_argument("--max-frames", type=int, help="Limit timestamps per camera for a smoke test.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate calibration and data without writing or training.")
    parser.add_argument("--evaluate-joint-axis", action="store_true", help="Compare the trained joint axis to GT and report error in degrees.")
    parser.add_argument("--visualize-results", action="store_true",
                        help="Make all-camera videos with estimated axes; also evaluate joint GT when available.")
    parser.add_argument("--reuse-tracking", action="store_true", help="Reuse matching prepared camera sequences; recompute ICP if poses have missing timestamp slots.")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip successful jobs with final checkpoint, exports and any requested GT reports/videos present.")
    args = parser.parse_args()
    if args.iterations < 1 or args.frame_stride < 1 or (args.max_frames is not None and args.max_frames < 2):
        parser.error("Iterations/stride must be positive and max-frames must be at least two")
    dataset_root = args.datasets.expanduser().resolve()
    outputs_root = args.outputs_root.expanduser().resolve()
    scenes = discover_jobs(dataset_root)
    selected_names = args.objects or ([args.object] if args.object else None)
    if selected_names:
        available = {scene.name: scene for scene in scenes}
        missing = [name for name in selected_names if name not in available]
        if missing:
            parser.error(f"Raw objects not found: {', '.join(missing)}")
        if len(set(selected_names)) != len(selected_names):
            parser.error("Object selection contains duplicate names")
        scenes = [available[name] for name in selected_names]
    print(f"Found {len(scenes)} multi-view object jobs; episode_000000 is excluded", flush=True)
    print("Experimental multiview adapter: independent camera segmentations and one-view motion "
          "initialization. Known part-label errors under occlusion; training completion does not "
          "validate reconstruction quality. See docs/official_pipeline_review.md.", flush=True)
    work = outputs_root / "work"
    input_root = work / "input"
    prepared_root = work / "datasets"
    configs = work / "configs"
    log_root = outputs_root / "batch_logs"
    for directory in (input_root, prepared_root, configs, log_root):
        if not directory.resolve().is_relative_to(outputs_root):
            raise ValueError(f"Working directory escapes the output target: {directory}")
    if any(outputs_root == scene or outputs_root.is_relative_to(scene) for scene in scenes):
        raise ValueError("The output target must be outside the raw source object folders")
    print(f"All generated data and runtime caches: {outputs_root}", flush=True)
    existing_results = []
    if (selected_names or args.skip_completed) and (log_root / "summary.json").is_file():
        existing_results = json.loads((log_root / "summary.json").read_text())
    records = {row["job"]: row for row in existing_results}
    jobs = []
    skipped = 0
    for scene in scenes:
        evaluate_axis = args.evaluate_joint_axis or (
            args.visualize_results and (scene / "pointcloud/joint_metadata.json").is_file())
        name = f"{scene.name}_multiview"
        if args.max_frames is not None or args.frame_stride != 1:
            name += f"_f{args.max_frames or 'all'}_s{args.frame_stride}"
        if args.skip_completed and result_is_complete(records.get(name), outputs_root / name / "splatfacto-art",
                                                      args.iterations, evaluate_axis, args.visualize_results):
            print(f"SKIP {name}: completed results verified", flush=True)
            skipped += 1
            continue
        if evaluate_axis:
            load_gt_axis(scene)
        cameras = sorted(p.name for p in (scene / "rgb").iterdir() if p.is_dir())
        if len(cameras) < 2:
            raise ValueError(f"Multi-view training requires at least two cameras: {scene}")
        poses = calibration_poses(scene)
        if any(camera not in poses for camera in cameras):
            raise ValueError(f"Missing COLMAP extrinsics in {scene}")
        options = []
        source_stems = None
        for camera in cameras:
            camera_name = f"{name}_{camera.replace('_', '')}"
            out = input_root / camera_name
            command = ["--input", str(scene), "--camera", camera, "--output", str(out),
                       "--write-config", str(configs / f"{camera_name}.yaml"),
                       "--dataset-root", str(prepared_root), "--outputs-root", str(outputs_root),
                       "--frame-stride", str(args.frame_stride), "--dry-run"]
            if args.max_frames is not None:
                command.extend(["--max-frames", str(args.max_frames)])
            opt = conversion_args(command)
            convert(opt)
            triplets, _ = build_triplets(scene, camera, opt.depth_source, opt.allow_missing_mask)
            stems = [frame.stem for frame in triplets[::args.frame_stride][:args.max_frames]]
            if source_stems is not None and stems != source_stems:
                raise ValueError(f"Unsynchronized camera timestamps in {scene}: {camera}")
            source_stems = stems
            if out.exists():
                provenance = out / "metadata/freeart_conversion.json"
                if not provenance.exists():
                    raise ValueError(f"Refusing to replace an unrelated input: {out}")
                meta = json.loads(provenance.read_text())
                old_scene = Path(meta["scene_root"]).resolve()
                rebuild_moved_source = (not args.reuse_tracking and not old_scene.exists()
                                        and old_scene.name == scene.name)
                if (old_scene != scene and not rebuild_moved_source) or meta["selected_camera"] != camera:
                    raise ValueError(f"Prepared input belongs to another source: {out}")
                if old_scene != scene:
                    print(f"Source path changed: {old_scene} -> {scene}; rebuilding prepared data.", flush=True)
                opt.overwrite = True
            options.append((camera, camera_name, opt))
        print(f"  {name}: {len(cameras)} cameras will train one shared model", flush=True)
        jobs.append((scene, name, options))
    print(f"Batch selection: {len(jobs)} jobs to run, {skipped} completed jobs skipped", flush=True)
    if args.dry_run:
        print("Dry run complete: no files written and no training started.")
        return
    if not jobs:
        print("All selected jobs are already complete; no training started.")
        return

    configure_runtime(outputs_root)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; run in the freeartgs environment outside the sandbox")
    configs.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    results = []
    previous_results = []
    if selected_names or args.skip_completed:
        selected_jobs = {name for _, name, _ in jobs}
        previous_results = [row for row in existing_results
                            if row["job"] not in selected_jobs]
    for scene, name, options in jobs:
        log_path = log_root / f"{name}.log"
        print(f"TRAINING {name} — log: {log_path}", flush=True)
        stop_batch = False
        try:
            require_source(scene)
            ensure_free_space(outputs_root)
            with log_path.open("a" if args.reuse_tracking else "w") as log:
                camera_data = {}
                for camera, camera_name, opt in options:
                    require_source(scene)
                    ensure_free_space(outputs_root)
                    data = prepared_root / camera_name
                    provenance = data / "metadata/freeart_conversion.json"
                    reuse = args.reuse_tracking and provenance.is_file()
                    if reuse:
                        meta = json.loads(provenance.read_text())
                        triplets, _ = build_triplets(scene, camera, opt.depth_source, opt.allow_missing_mask)
                        expected = [frame.stem for frame in triplets[::opt.frame_stride][:opt.max_frames]]
                        if (Path(meta["scene_root"]).resolve() != scene or meta["selected_camera"] != camera
                                or [row["source_stem"] for row in meta["frame_map"]] != expected):
                            raise ValueError(f"Prepared tracking does not match selected raw frames: {data}")
                        print(f"REUSING tracking: {data}", flush=True)
                        maybe_write_config(opt, camera_name, load_camera_entry(scene, camera))
                        rerun_icp = False
                        for part in (0, 1):
                            path = data / f"global_poses_{part}.npy"
                            poses = np.load(path) if path.is_file() else np.empty(0)
                            rerun_icp |= poses.shape != (len(expected), 4, 4) or not np.isfinite(poses).all()
                    else:
                        opt.dry_run = False
                        convert(opt)
                        run_stage(opt.write_config, log, "--preprocess_only")
                        run_stage(opt.write_config, log, "--tracking_only")
                        rerun_icp = True
                    if rerun_icp:
                        subprocess.run(["bash", "pipelines/icp.sh", "--config", str(opt.write_config)],
                                       cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    camera_data[camera] = data
                ensure_free_space(outputs_root)
                require_source(scene)
                dataset = prepared_root / name
                prepare_multiview(scene, camera_data, dataset)
                config = yaml.safe_load(options[0][2].write_config.read_text())
                config["paths"]["object_name"] = name
                config["multiview"] = {"enabled": True}
                for stage in ("train_parts", "twopart_blend", "splatfacto_art"):
                    config[stage].update(max_num_iterations=args.iterations, camera_optimizer_mode="off")
                config_path = configs / f"{name}.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                run_stage(config_path, log, "--reconstruction_only")
            result = dict(job=name, status="passed", training_completed=True,
                          reconstruction_quality_validated=False, cameras=len(options), log=str(log_path))
            final_output = outputs_root / name / "splatfacto-art"
            evaluate_axis = args.evaluate_joint_axis or (
                args.visualize_results and (scene / "pointcloud/joint_metadata.json").is_file())
            if evaluate_axis:
                report = evaluate_joint_axis(scene, dataset, final_output)
                result.update(joint_axis_error_deg=report["joint_axis_error_deg"],
                              joint_axis_evaluation=str(final_output / "joint_axis_error.json"))
                for key in ("joint_type_correct", "joint_state_mae", "joint_state_unit", "evaluated_timestamps"):
                    if key in report:
                        result[key] = report[key]
                print(f"{name}: joint axis error = {report['joint_axis_error_deg']:.6f} degrees", flush=True)
                if "joint_state_mae" in report:
                    print(f"{name}: relative joint-state MAE = {report['joint_state_mae']:.6f} "
                          f"{report['joint_state_unit']}; joint type correct: {report['joint_type_correct']}", flush=True)
            if args.visualize_results:
                try:
                    with log_path.open("a") as log:
                        subprocess.run([sys.executable, str(REPO_ROOT / "evaluate/visualize_local_comparison.py"),
                                        "--scene", str(scene), "--dataset", str(dataset),
                                        "--results", str(final_output)], cwd=REPO_ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    basename = "motion_vs_gt" if evaluate_axis else "motion_axis_debug"
                    result["motion_comparison"] = str(final_output / "comparison" / f"{basename}.mp4")
                except Exception as error:
                    result["visualization_error"] = str(error)
                    stop_batch = storage_failure(error, log_path)
                    print(f"Comparison failed for {name}: {error}; see {log_path}", file=sys.stderr, flush=True)
            results.append(result)
        except Exception as error:
            stop_batch = not scene.is_dir() or storage_failure(error, log_path)
            print(f"FAILED {name}: {error}; see {log_path}", file=sys.stderr, flush=True)
            results.append(dict(job=name, status="failed", error=str(error), log=str(log_path)))
        write_summary(log_root / "summary.json", previous_results + results)
        if stop_batch:
            print("Stopping the batch after a source or storage failure; later objects were not started.", file=sys.stderr, flush=True)
            sys.exit(1)
    failed = sum(row["status"] == "failed" for row in results)
    print(f"Finished: {len(results) - failed} training jobs completed, {failed} failed. Logs: {log_root}")
    errors = [row["joint_axis_error_deg"] for row in results if "joint_axis_error_deg" in row]
    if errors:
        print(f"Mean joint axis error over {len(errors)} evaluated objects: {sum(errors) / len(errors):.6f} degrees")
    if failed or any("visualization_error" in row for row in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
