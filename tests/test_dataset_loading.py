import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

from samuele.convert_standard_to_freeart import camera_intrinsics, convert, load_camera_entry
from pipelines.train_local_datasets import discover_jobs
from pipelines import train_local_datasets as batch
from utils.utils import load_images
from preprocessing.prepare_multiview import calibration_poses, prepare_multiview


class DatasetLoadingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for patcher in (patch.object(batch, "DEFAULT_OUTPUTS_ROOT", self.root / "outputs"),
                        patch("utils.local_storage.MIN_FREE_BYTES", 0),
                        patch.dict(os.environ), patch.object(tempfile, "tempdir", tempfile.tempdir)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.scene = self.root / "scene"
        self.camera = dict(name="cam_000", width=8, height=6, fov_y_deg=45,
                           fx=None, fy=None, cx=None, cy=None, far=10)
        (self.scene / "metadata").mkdir(parents=True)
        (self.scene / "metadata/cameras.json").write_text(json.dumps({"cameras": [self.camera]}))
        for folder in ("rgb", "depth_npy", "mask"):
            (self.scene / folder / "cam_000").mkdir(parents=True)
        for i in range(2):
            stem = f"frame_{i:06d}"
            Image.fromarray(np.full((6, 8, 3), 80 + i, dtype=np.uint8)).save(self.scene / "rgb/cam_000" / f"{stem}.png")
            np.save(self.scene / "depth_npy/cam_000" / f"{stem}.npy", np.full((6, 8), 4.0 + i, dtype=np.float32))
            Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(self.scene / "mask/cam_000" / f"{stem}.png")
        self.args = argparse.Namespace(input=self.scene, output=self.root / "converted",
            camera="cam_000", depth_source="auto", depth_png_scale=.001,
            frame_stride=1, max_frames=None, start_index=0, name_width=5,
            rgb_format="png", mask_threshold=127, allow_missing_mask=False,
            min_depth=0, max_depth=None, write_config=self.root / "config.yaml",
            original_data_dir=None, dataset_root="datasets", outputs_root="outputs",
            overwrite=False, dry_run=False)

    def run_conversion(self):
        with contextlib.redirect_stdout(io.StringIO()):
            convert(self.args)

    def add_second_camera(self):
        camera = dict(self.camera, name="cam_001")
        (self.scene / "metadata/cameras.json").write_text(json.dumps({"cameras": [self.camera, camera]}))
        for folder in ("rgb", "depth_npy", "mask"):
            shutil.copytree(self.scene / folder / "cam_000", self.scene / folder / "cam_001")
        sparse = self.scene / "colmap/sparse/0"
        sparse.mkdir(parents=True)
        (sparse / "cameras.txt").write_text("7 PINHOLE 8 6 12 13 3 2\n42 PINHOLE 8 6 15 16 4 3\n")
        (sparse / "images.txt").write_text(
            "1 1 0 0 0 0 0 0 7 cam_000/frame_000000.png\n\n"
            "2 1 0 0 0 -0.2 0 0 42 cam_001/frame_000000.png\n\n")

    def test_fov_and_explicit_intrinsics(self):
        k = camera_intrinsics(self.camera)
        self.assertAlmostEqual(k[1, 1], 6 / (2 * np.tan(np.pi / 8)))
        np.testing.assert_allclose(k[:2, 2], [4, 3])
        camera = dict(self.camera, fx=20, fy=21, cx=2, cy=3)
        np.testing.assert_allclose(camera_intrinsics(camera), [[20, 0, 2], [0, 21, 3], [0, 0, 1]])

    def test_colmap_calibration_matches_image_camera_id(self):
        sparse = self.scene / "colmap/sparse/0"
        sparse.mkdir(parents=True)
        (sparse / "cameras.txt").write_text(
            "# cameras\n7 PINHOLE 8 6 90 91 4 3\n42 PINHOLE 8 6 12 13 3 2\n")
        (sparse / "images.txt").write_text(
            "# images\n1 1 0 0 0 0 0 0 7 cam_001/frame_000000.png\n\n"
            "2 1 0 0 0 0 0 0 42 cam_000/frame_000000.png\n0 0 -1\n")
        entry = load_camera_entry(self.scene, "cam_000")
        self.assertEqual(entry["colmap_camera_id"], 42)
        np.testing.assert_allclose(camera_intrinsics(entry), [[12, 0, 3], [0, 13, 2], [0, 0, 1]])
        self.assertEqual(entry["intrinsics_source"], str(sparse / "cameras.txt"))

    def test_batch_discovery_avoids_duplicate_prepared_sequences(self):
        self.run_conversion()
        for part in (0, 1):
            (self.args.output / f"transforms_{part}.json").write_text('{}')
        real = self.root / "episode_000000"
        shutil.copytree(self.scene, real)
        for part in (0, 1):
            (real / f"transforms_{part}.json").write_text('{}')
        jobs = discover_jobs(self.root)
        self.assertEqual(jobs, [self.scene])

    def test_batch_dry_run_writes_nothing(self):
        self.add_second_camera()
        script = Path(__file__).resolve().parents[1] / "pipelines/train_local_datasets.py"
        target = self.root / "Extreme SSD/freeart"
        result = subprocess.run([sys.executable, str(script), "--datasets", str(self.root),
                                 "--outputs-root", str(target), "--all-cameras", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Found 1 multi-view object jobs", result.stdout)
        self.assertIn("2 cameras will train one shared model", result.stdout)
        self.assertIn("no files written", result.stdout)
        self.assertFalse((self.root / "converted").exists())
        self.assertFalse(target.exists())

    def test_batch_stops_after_subprocess_disk_failure(self):
        self.add_second_camera()
        shutil.copytree(self.scene, self.root / "scene2")

        def disk_failure(command, *, stdout, **kwargs):
            stdout.write("OSError: [Errno 28] No space left on device\n")
            stdout.flush()
            raise subprocess.CalledProcessError(1, command)

        with patch.object(batch, "REPO_ROOT", self.root), \
             patch.object(sys, "argv", ["batch", "--datasets", str(self.root), "--iterations", "10"]), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.object(batch.subprocess, "run", side_effect=disk_failure) as run, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                batch.main()
        self.assertEqual(exit_result.exception.code, 1)
        self.assertEqual(run.call_count, 1)
        summary = json.loads((self.root / "outputs/batch_logs/summary.json").read_text())
        self.assertEqual([row["job"] for row in summary], ["scene_multiview"])
        self.assertEqual(summary[0]["status"], "failed")
        self.assertFalse((self.root / "outputs/batch_logs/scene2_multiview.log").exists())

    def test_moved_source_rebuilds_existing_input_without_reusing_tracking(self):
        self.add_second_camera()
        self.args.output = self.root / "outputs/work/input/scene_multiview_cam000"
        self.run_conversion()
        backup = self.root / "Backup SSD/raw"
        backup.mkdir(parents=True)
        relocated = backup / self.scene.name
        self.scene.rename(relocated)
        with patch.object(batch, "REPO_ROOT", self.root), \
             patch.object(sys, "argv", ["batch", "--datasets", str(backup), "--object", "scene", "--iterations", "10"]), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.object(batch, "prepare_multiview", return_value={}), \
             patch.object(batch.subprocess, "run", return_value=subprocess.CompletedProcess(["bash"], 0)) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            batch.main()
        self.assertEqual(run.call_count, 7)
        meta = json.loads((self.args.output / "metadata/freeart_conversion.json").read_text())
        self.assertEqual(Path(meta["scene_root"]), relocated)
        self.assertTrue(all(Path(row["rgb"]).is_relative_to(relocated) for row in meta["frame_map"]))
        self.assertFalse(self.scene.exists())
        self.assertTrue((relocated / "rgb/cam_000/frame_000000.png").is_file())

    def test_existing_other_source_is_still_protected(self):
        self.add_second_camera()
        self.args.output = self.root / "outputs/work/input/scene_multiview_cam000"
        self.run_conversion()
        backup = self.root / "backup"
        shutil.copytree(self.scene, backup / self.scene.name)
        with patch.object(batch, "REPO_ROOT", self.root), \
             patch.object(sys, "argv", ["batch", "--datasets", str(backup), "--dry-run"]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "another source"):
                batch.main()
        self.assertTrue(self.scene.exists())

    def test_batch_stops_if_source_disappears_after_preflight(self):
        self.add_second_camera()
        shutil.copytree(self.scene, self.root / "scene2")

        def move_source(command, **kwargs):
            if self.scene.exists():
                self.scene.rename(self.root / "moved_scene")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(batch, "REPO_ROOT", self.root), \
             patch.object(sys, "argv", ["batch", "--datasets", str(self.root), "--iterations", "10"]), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.object(batch.subprocess, "run", side_effect=move_source) as run, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                batch.main()
        self.assertEqual(exit_result.exception.code, 1)
        self.assertEqual(run.call_count, 3)
        summary = json.loads((self.root / "outputs/batch_logs/summary.json").read_text())
        self.assertEqual([row["job"] for row in summary], ["scene_multiview"])
        self.assertIn("Raw source disappeared after validation", summary[0]["error"])
        self.assertFalse((self.root / "outputs/batch_logs/scene2_multiview.log").exists())

    def test_missing_sources_report_the_path_and_fail_early(self):
        missing = self.root / "missing"
        with self.assertRaisesRegex(FileNotFoundError, "Pass --datasets"):
            discover_jobs(missing)
        self.args.input = missing
        with self.assertRaisesRegex(FileNotFoundError, "missing or disconnected"):
            self.run_conversion()

    def test_batch_continues_after_training_failure(self):
        self.add_second_camera()
        shutil.copytree(self.scene, self.root / "scene2")
        real = self.root / "episode_000000"
        real.mkdir()
        for part in (0, 1):
            (real / f"transforms_{part}.json").write_text('{}')
        (self.root / "configs").mkdir()
        shutil.copy2(batch.REPO_ROOT / "configs/freeart-21.yaml", self.root / "configs/freeart-21.yaml")
        calls = [subprocess.CalledProcessError(1, ["bash"])] + [subprocess.CompletedProcess(["bash"], 0)] * 7
        with patch.object(batch, "REPO_ROOT", self.root), \
             patch.object(sys, "argv", ["batch", "--datasets", str(self.root), "--iterations", "10"]), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.object(batch, "prepare_multiview", return_value={}), \
             patch.object(batch.subprocess, "run", side_effect=calls) as run, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                batch.main()
        self.assertEqual(exit_result.exception.code, 1)
        self.assertEqual(run.call_count, 8)
        summary = json.loads((self.root / "outputs/batch_logs/summary.json").read_text())
        self.assertEqual([r["status"] for r in summary], ["failed", "passed"])

    def test_conversion_config_and_tracking_loader(self):
        self.run_conversion()
        config = yaml.safe_load(self.args.write_config.read_text())
        self.assertEqual(config["tracking"]["max_depth"], 10)
        self.assertEqual(config["tracking"]["min_trajectory_count"], 2000)
        self.assertEqual(config["preprocess"]["pcd_max_depth"], 10)
        self.assertFalse(config["evaluation"]["enabled"])
        for section in ("train_parts", "twopart_blend", "splatfacto_art"):
            self.assertIn("max_num_iterations", config[section])
            self.assertEqual(config[section]["quaternion_order"], "xyzw")
        (self.args.output / "rgb").rename(self.args.output / "images_ori")
        with contextlib.redirect_stdout(io.StringIO()):
            data = load_images(str(self.args.output), append_reversed=True)
        self.assertEqual(tuple(data["rgbs_tensor"].shape), (1, 3, 3, 6, 8))
        self.assertEqual([float(d[0, 0]) for d in data["depths"]], [4, 5, 4])
        (self.args.output / "depth/00001.npz").rename(self.args.output / "depth/00002.npz")
        with self.assertRaisesRegex(ValueError, "alignment"):
            load_images(str(self.args.output))

    def test_batch_reports_gt_axis_error_after_training(self):
        self.add_second_camera()
        (self.scene / "pointcloud").mkdir()
        (self.scene / "pointcloud/joint_metadata.json").write_text(json.dumps(
            dict(coordinate_frame="render_world", axis_vector=[0, 0, 1])))
        (self.root / "configs").mkdir()
        shutil.copy2(batch.REPO_ROOT / "configs/freeart-21.yaml", self.root / "configs/freeart-21.yaml")
        with patch.object(batch, "REPO_ROOT", self.root), \
             patch.object(sys, "argv", ["batch", "--datasets", str(self.root), "--iterations", "10",
                                       "--evaluate-joint-axis"]), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.object(batch, "prepare_multiview", return_value={}), \
             patch.object(batch.subprocess, "run", return_value=subprocess.CompletedProcess(["bash"], 0)), \
             patch.object(batch, "evaluate_joint_axis", return_value={"joint_axis_error_deg": 12.5}) as evaluate, \
             contextlib.redirect_stdout(io.StringIO()):
            batch.main()
        evaluate.assert_called_once_with(self.scene, self.root / "outputs/work/datasets/scene_multiview",
                                         self.root / "outputs/scene_multiview/splatfacto-art")
        summary = json.loads((self.root / "outputs/batch_logs/summary.json").read_text())
        self.assertEqual(summary[0]["status"], "passed")
        self.assertEqual(summary[0]["joint_axis_error_deg"], 12.5)

    def test_batch_visualizes_real_data_without_requiring_joint_gt(self):
        self.add_second_camera()
        with patch.object(batch, "REPO_ROOT", self.root), \
             patch.object(sys, "argv", ["batch", "--datasets", str(self.root), "--iterations", "10",
                                       "--visualize-results"]), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.object(batch, "prepare_multiview", return_value={}), \
             patch.object(batch.subprocess, "run", return_value=subprocess.CompletedProcess(["bash"], 0)) as run, \
             patch.object(batch, "evaluate_joint_axis") as evaluate, \
             contextlib.redirect_stdout(io.StringIO()):
            batch.main()
        evaluate.assert_not_called()
        final = self.root / "outputs/scene_multiview/splatfacto-art"
        self.assertIn(str(final), run.call_args.args[0])
        summary = json.loads((self.root / "outputs/batch_logs/summary.json").read_text())
        self.assertEqual(summary[0]["motion_comparison"], str(final / "comparison/motion_axis_debug.mp4"))
        self.assertNotIn("joint_axis_error_deg", summary[0])

    def make_completed_result(self, name="scene_multiview"):
        outputs = self.root / "outputs"
        final = outputs / name / "splatfacto-art"
        for filename in ("config.yml", "nerfstudio_models/step-000000009.ckpt", "object_3dgs.ply",
                         "3dgs_part_weight.txt", "comparison/motion_vs_gt.mp4", "comparison/motion_vs_gt.gif"):
            path = final / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"saved artifact")
        for filename, data in (("articulation.json", {"frames": [{"frame_index": 0}]}),
                               ("transforms_train.json", [{"file_path": "00000.png"}]),
                               ("joint_axis_error.json", {"joint_axis_error_deg": 1.5}),
                               ("comparison/comparison_info.json", {"timestamps": 2})):
            (final / filename).write_text(json.dumps(data))
        log = outputs / "batch_logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Existing successful training log\n")
        row = dict(job=name, status="passed", training_completed=True, cameras=2, log=str(log),
                   reconstruction_quality_validated=False, joint_axis_error_deg=1.5,
                   motion_comparison=str(final / "comparison/motion_vs_gt.mp4"))
        (log.parent / "summary.json").write_text(json.dumps([row], indent=2))
        return final, row

    def test_completion_requires_final_artifacts_and_requested_outputs(self):
        final, record = self.make_completed_result()
        self.assertTrue(batch.result_is_complete(record, final, 10, True, True))
        self.assertFalse(batch.result_is_complete(dict(record, status="failed"), final, 10))
        self.assertFalse(batch.result_is_complete(dict(record, training_completed=False), final, 10))
        self.assertFalse(batch.result_is_complete(record, final, 20))
        self.assertFalse(batch.result_is_complete(dict(record, visualization_error="failed"), final, 10, True, True))
        for filename in ("nerfstudio_models/step-000000009.ckpt", "object_3dgs.ply",
                         "joint_axis_error.json", "comparison/motion_vs_gt.mp4"):
            with self.subTest(filename=filename):
                path = final / filename
                data = path.read_bytes()
                path.unlink()
                self.assertFalse(batch.result_is_complete(record, final, 10, True, True))
                path.write_bytes(b"")
                self.assertFalse(batch.result_is_complete(record, final, 10, True, True))
                path.write_bytes(data)
        (final / "articulation.json").write_text('{"frames": [')
        self.assertFalse(batch.result_is_complete(record, final, 10))

    def test_completion_accepts_estimated_axis_debug_without_gt_report(self):
        final, record = self.make_completed_result()
        (final / "joint_axis_error.json").unlink()
        for suffix in ("mp4", "gif"):
            (final / f"comparison/motion_vs_gt.{suffix}").rename(final / f"comparison/motion_axis_debug.{suffix}")
        self.assertTrue(batch.result_is_complete(record, final, 10, False, True))
        self.assertFalse(batch.result_is_complete(record, final, 10, True, True))

    def test_all_completed_jobs_skip_validation_gpu_and_keep_files_unchanged(self):
        final, record = self.make_completed_result()
        # A completed object does not need its raw frames revalidated.
        (self.scene / "depth_npy/cam_000/frame_000001.npy").unlink()
        summary = self.root / "outputs/batch_logs/summary.json"
        summary_before = summary.read_bytes()
        log_before = Path(record["log"]).read_bytes()
        argv = ["batch", "--datasets", str(self.root), "--object", "scene", "--iterations", "10",
                "--skip-completed", "--evaluate-joint-axis", "--visualize-results"]
        with patch.object(batch, "REPO_ROOT", self.root), patch.object(sys, "argv", argv), \
             patch.object(batch, "configure_runtime") as runtime, \
             patch.object(batch, "convert") as convert_mock, \
             patch.object(batch.subprocess, "run") as run, \
             patch("torch.cuda.is_available", return_value=False) as cuda, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            batch.main()
        self.assertIn("SKIP scene_multiview", output.getvalue())
        for mock in (runtime, convert_mock, run, cuda):
            mock.assert_not_called()
        wrapper = Path(__file__).resolve().parents[1] / "pipelines/train_local_datasets.sh"
        result = subprocess.run(["bash", str(wrapper), *argv[1:], "--outputs-root", str(self.root / "outputs")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("All selected jobs are already complete", result.stdout)
        self.assertEqual(summary.read_bytes(), summary_before)
        self.assertEqual(Path(record["log"]).read_bytes(), log_before)
        self.assertFalse((self.root / "outputs/work").exists())
        self.assertFalse((self.root / "outputs/.runtime").exists())

    def test_skip_completed_preserves_record_and_runs_remaining_object(self):
        self.add_second_camera()
        shutil.copytree(self.scene, self.root / "scene2")
        final, record = self.make_completed_result()
        with patch.object(batch, "REPO_ROOT", self.root), \
             patch.object(sys, "argv", ["batch", "--datasets", str(self.root), "--iterations", "10", "--skip-completed"]), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.object(batch, "prepare_multiview", return_value={}), \
             patch.object(batch.subprocess, "run", return_value=subprocess.CompletedProcess(["bash"], 0)) as run, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            batch.main()
        self.assertEqual(run.call_count, 7)
        self.assertIn("1 jobs to run, 1 completed jobs skipped", output.getvalue())
        summary = json.loads((self.root / "outputs/batch_logs/summary.json").read_text())
        self.assertEqual(summary[0], record)
        self.assertEqual(summary[1]["job"], "scene2_multiview")
        self.assertEqual(summary[1]["status"], "passed")
        self.assertEqual(Path(record["log"]).read_text(), "Existing successful training log\n")
        self.assertFalse((self.root / "outputs/work/input/scene_multiview_cam000").exists())

    def test_batch_selected_objects_custom_outputs_and_videos(self):
        self.add_second_camera()
        (self.scene / "pointcloud").mkdir()
        (self.scene / "pointcloud/joint_metadata.json").write_text(json.dumps(
            dict(coordinate_frame="render_world", axis_vector=[0, 0, 1])))
        shutil.copytree(self.scene, self.root / "scene2")
        shutil.copytree(self.scene, self.root / "unselected")
        (self.root / "unselected/pointcloud/joint_metadata.json").unlink()
        (self.root / "configs").mkdir()
        shutil.copy2(batch.REPO_ROOT / "configs/freeart-21.yaml", self.root / "configs/freeart-21.yaml")
        outputs = self.root / "Extreme SSD/results/freeart"
        (outputs / "batch_logs").mkdir(parents=True)
        previous = {"job": "unselected_multiview", "status": "passed"}
        (outputs / "batch_logs/summary.json").write_text(json.dumps([previous]))
        completed = subprocess.CompletedProcess(["bash"], 0)
        # Seven training/frontend calls per object, followed by video generation.
        calls = [completed] * 7 + [subprocess.CalledProcessError(1, ["ffmpeg"])] + [completed] * 8
        with patch.object(batch, "REPO_ROOT", self.root), \
             patch.object(sys, "argv", ["batch", "--datasets", str(self.root),
                                       "--objects", "scene2", "scene", "--outputs-root", str(outputs),
                                       "--iterations", "10", "--visualize-results"]), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.object(batch, "prepare_multiview", return_value={}), \
             patch.object(batch.subprocess, "run", side_effect=calls) as run, \
             patch.object(batch, "evaluate_joint_axis", return_value={"joint_axis_error_deg": 12.5}) as evaluate, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                batch.main()
        self.assertEqual(exit_result.exception.code, 1)
        self.assertEqual(run.call_count, 16)
        self.assertEqual(evaluate.call_count, 2)
        summary = json.loads((outputs / "batch_logs/summary.json").read_text())
        self.assertEqual([r["job"] for r in summary], ["unselected_multiview", "scene2_multiview", "scene_multiview"])
        self.assertEqual(summary[0], previous)
        self.assertTrue(summary[1]["training_completed"])
        self.assertIn("visualization_error", summary[1])
        final_output = outputs / "scene_multiview/splatfacto-art"
        self.assertEqual(summary[2]["motion_comparison"], str(final_output / "comparison/motion_vs_gt.mp4"))
        self.assertEqual(summary[2]["joint_axis_evaluation"], str(final_output / "joint_axis_error.json"))
        self.assertIn(str(final_output), run.call_args.args[0])
        config = yaml.safe_load((outputs / "work/configs/scene_multiview.yaml").read_text())
        self.assertEqual(config["paths"]["output_root"], str(outputs))
        self.assertEqual(config["paths"]["dataset_root"], str(outputs / "work/datasets"))
        self.assertEqual(config["paths"]["original_data_dir"], str(outputs / "work/input"))
        self.assertFalse((self.root / "input").exists())
        self.assertFalse((self.root / "scene_multiview").exists())
        self.assertFalse((self.root / "configs/local_datasets").exists())
        self.assertTrue((outputs / "work/input/scene_multiview_cam000/depth/00000.npz").exists())
        self.assertFalse((self.root / "outputs").exists())

    def test_batch_rejects_unknown_and_duplicate_selections_before_writing(self):
        for names in (["scene", "typo"], ["scene", "scene"], ["episode_000000"]):
            with self.subTest(names=names), patch.object(batch, "REPO_ROOT", self.root), \
                 patch.object(sys, "argv", ["batch", "--datasets", str(self.root), "--objects", *names]), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_result:
                    batch.main()
                self.assertEqual(exit_result.exception.code, 2)
        self.assertFalse((self.root / "outputs").exists())

    def test_dry_run_validates_and_writes_nothing(self):
        self.args.dry_run = True
        self.run_conversion()
        self.assertFalse(self.args.output.exists())
        np.save(self.scene / "depth_npy/cam_000/frame_000001.npy", np.zeros((6, 8), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "foreground depth"):
            self.run_conversion()

    def test_unmatched_frames_are_rejected(self):
        (self.scene / "depth_npy/cam_000/frame_000001.npy").unlink()
        with self.assertRaisesRegex(ValueError, "stems differ"):
            self.run_conversion()

    def test_input_is_protected(self):
        self.args.output = self.scene
        self.args.overwrite = True
        with self.assertRaisesRegex(ValueError, "replace the input"):
            self.run_conversion()
        self.assertTrue((self.scene / "metadata/cameras.json").exists())

    def test_preprocessing_rejects_same_input_and_output(self):
        self.run_conversion()
        script = Path(__file__).resolve().parents[1] / "preprocessing/segment.py"
        result = subprocess.run([sys.executable, str(script), "--input_dir", str(self.args.output),
                                 "--output_dir", str(self.root), "--skip_segment"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input and output must differ", result.stderr)
        self.assertTrue((self.args.output / "rgb/00000.png").exists())

    def test_black_and_white_seed_colors_remain_trainable(self):
        import torch
        from nerfstudio.data.scene_box import SceneBox
        from nerfstudio.models.splatfacto import SplatfactoModelConfig

        points = torch.zeros((4, 3))
        colors = torch.tensor([[0, 0, 0], [255, 255, 255], [127, 255, 0], [255, 0, 127]], dtype=torch.uint8)
        with patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self):
            model = SplatfactoModelConfig(sh_degree=0).setup(
                scene_box=SceneBox(torch.tensor([[-2., -2., -2.], [2., 2., 2.]])),
                num_train_data=2, metadata={}, device="cpu", seed_points=(points, colors))
        self.assertTrue(torch.isfinite(model.features_dc).all())
        self.assertTrue(torch.isfinite(model.scales).all())
        model.colors.sum().backward()
        self.assertTrue(torch.isfinite(model.features_dc.grad).all())
        self.assertTrue((model.features_dc.grad > 0).all())

    def make_tracked_multiview(self):
        self.add_second_camera()
        datasets = {}
        for camera, moving in (("cam_000", 1), ("cam_001", 0)):
            self.args.camera = camera
            self.args.output = self.root / f"tracked_{camera}"
            self.run_conversion()
            data = self.args.output
            (data / "rgb").rename(data / "images_ori")
            shutil.copytree(data / "images_ori", data / "images")
            for part in (0, 1):
                (data / f"mask_{part}").mkdir()
                poses = np.tile(np.eye(4), (2, 1, 1))
                if part == moving:
                    poses[1, 0, 3] = .3
                np.save(data / f"global_poses_{part}.npy", poses)
                for timestamp in range(2):
                    Image.fromarray(np.full((6, 8), 255 if part == moving else 0, np.uint8)).save(
                        data / f"mask_{part}/{timestamp:05d}.png")
            datasets[camera] = data
        return datasets

    def test_multiview_preserves_calibration_and_shares_motion(self):
        import torch
        from nerfstudio.models.twopart_blend import TwoPartBlendModel

        datasets = self.make_tracked_multiview()
        output = self.root / "multiview"
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = prepare_multiview(self.scene, datasets, output)
        self.assertEqual([f["time_index"] for f in manifest["frames"]], [0, 0, 1, 1])
        self.assertEqual([f["fl_x"] for f in manifest["frames"]], [12, 15, 12, 15])
        parts = [json.loads((output / f"transforms_{p}.json").read_text())["frames"] for p in (0, 1)]
        c0, c1 = [np.array([f["transform_matrix"] for f in frames]) for frames in parts]
        calibrations = calibration_poses(self.scene)
        np.testing.assert_allclose(c0, [calibrations[c] for c in ("cam_000", "cam_001")] * 2)
        object_poses = c0 @ np.linalg.inv(c1)
        np.testing.assert_allclose(object_poses[0], object_poses[1], atol=1e-6)
        np.testing.assert_allclose(object_poses[2], object_poses[3], atol=1e-6)
        self.assertAlmostEqual(object_poses[2, 0, 3], -.3)
        self.assertEqual(np.asarray(Image.open(output / "mask_1/00001.png")).min(), 255)

        from nerfstudio.data.dataparsers.part_dataparser import PartDataParserConfig
        parser = PartDataParserConfig(data=output, load_3D_points=False,
            orientation_method="none", center_poses=False, auto_scale_poses=False).setup()
        result = parser.get_dataparser_outputs("train")
        np.testing.assert_allclose(result.cameras.fx.flatten().numpy(), [12, 15, 12, 15])
        np.testing.assert_allclose(result.cameras.fy.flatten().numpy(), [13, 16, 13, 16])
        self.assertEqual(len(result.metadata["depth_filenames"]), 4)

        class PoseModel(torch.nn.Module):
            _init_object_poses = TwoPartBlendModel._init_object_poses

        model = PoseModel()
        model.motion_part = 1
        model._c2w0, model._c2w1 = torch.tensor(c0).float(), torch.tensor(c1).float()
        model.metadata = {"time_indices": [0, 0, 1, 1]}
        model.config = argparse.Namespace(obj_pose_opt_enabled=True)
        model._init_object_poses("cpu")
        self.assertEqual(model.obj_pose_adjustment.shape, (2, 6))
        np.testing.assert_allclose(model._T_obj_base.numpy(), object_poses[[0, 2]], atol=1e-6)
        model._c2w1[1, 0, 3] += .1
        with self.assertRaisesRegex(ValueError, "share the initial object pose"):
            model._init_object_poses("cpu")

    def test_multiview_rejects_unsynchronized_views_before_writing(self):
        datasets = self.make_tracked_multiview()
        path = datasets["cam_001"] / "metadata/freeart_conversion.json"
        metadata = json.loads(path.read_text())
        metadata["frame_map"][1]["source_stem"] = "different_timestamp"
        path.write_text(json.dumps(metadata))
        output = self.root / "multiview"
        with self.assertRaisesRegex(ValueError, "Unsynchronized"):
            prepare_multiview(self.scene, datasets, output)
        self.assertFalse(output.exists())

    def test_multiview_prefers_motion_with_better_timestamp_coverage(self):
        datasets = self.make_tracked_multiview()
        path = datasets["cam_000"] / "global_poses_1.npy"
        poses = np.load(path)
        poses[1, 0, 3] = 5
        np.save(path, poses)
        np.save(datasets["cam_000"] / "pose_valid_1.npy", [True, False])
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = prepare_multiview(self.scene, datasets, self.root / "multiview")
        self.assertEqual(manifest["reference_camera"], "cam_001")
        self.assertEqual(manifest["part_assignments"]["cam_000"]["valid_pose_counts"], [2, 1])

    def test_reuse_tracking_repairs_only_short_icp_and_preserves_other_results(self):
        datasets = self.make_tracked_multiview()
        prepared = self.root / "outputs/work/datasets"
        prepared.mkdir(parents=True)
        for camera, data in datasets.items():
            data.rename(prepared / f"scene_multiview_{camera.replace('_', '')}")
        data = prepared / "scene_multiview_cam000"
        np.save(data / "global_poses_0.npy", np.eye(4)[None])
        log_root = self.root / "outputs/batch_logs"
        log_root.mkdir(parents=True)
        (log_root / "summary.json").write_text(json.dumps([dict(job="other_multiview", status="passed")]))
        with patch.object(batch, "REPO_ROOT", self.root), \
             patch.object(sys, "argv", ["batch", "--datasets", str(self.root), "--object", "scene",
                                       "--iterations", "10", "--reuse-tracking"]), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.object(batch, "prepare_multiview", return_value={}), \
             patch.object(batch.subprocess, "run", return_value=subprocess.CompletedProcess(["bash"], 0)) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            batch.main()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][1], "pipelines/icp.sh")
        self.assertIn("--reconstruction_only", run.call_args_list[1].args[0])
        rows = json.loads((log_root / "summary.json").read_text())
        self.assertEqual([row["job"] for row in rows], ["other_multiview", "scene_multiview"])

    def test_multiview_seed_cloud_deduplicates_static_samples(self):
        import open3d as o3d
        from reconstruction.generate_ply import process_part

        datasets = self.make_tracked_multiview()
        output = self.root / "multiview"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_multiview(self.scene, datasets, output)
            path = output / "transforms_1.json"
            meta = json.loads(path.read_text())
            for i in (2, 3):
                meta["frames"][i]["transform_matrix"] = meta["frames"][i - 2]["transform_matrix"]
                shutil.copy2(output / f"depth/{i - 2:05d}.npz", output / f"depth/{i:05d}.npz")
            path.write_text(json.dumps(meta))
            cloud_path = process_part(output, part_id=1, pixel_stride=1)
        points = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)
        self.assertGreater(len(points), 48)
        self.assertLessEqual(len(points), 96)
        self.assertEqual(len(points), len(np.unique(points, axis=0)))

    def test_art_parser_uses_per_view_intrinsics_and_shared_joint(self):
        import torch
        from nerfstudio.data.dataparsers import splatfacto_art_dataparser as art

        datasets = self.make_tracked_multiview()
        output = self.root / "multiview"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_multiview(self.scene, datasets, output)
        blend = self.root / "blend"
        blend.mkdir()
        (blend / "object_3dgs.ply").touch()
        np.savetxt(blend / "3dgs_part_weight.txt", [.1, .9])
        frames = json.loads((output / "transforms_0.json").read_text())["frames"]
        (blend / "transforms_train.json").write_text(json.dumps([
            {"file_path": str(output / f["file_path"]), "transform": f["transform_matrix"]} for f in frames]))
        (blend / "estimated_joint.json").write_text(json.dumps(dict(
            joint_type="revolute", axis=[0, 0, 1], origin=[0, 0, 0],
            global_transform=np.eye(4).tolist(), joint_params=[0, .3])))
        config = art.SplatfactoArtDataParserConfig(data_dir=output, twopart_blend_dir=blend,
            orientation_method="none", center_poses=False, auto_scale_poses=False)
        gaussians = {"means": torch.zeros((2, 3)), "scales": torch.zeros((2, 3))}
        with patch.object(art, "load_3dgs_ply", return_value=gaussians), \
             patch.object(art, "apply_left_se3_to_gaussians", side_effect=lambda g, *args: g):
            result = config.setup().get_dataparser_outputs("train")
        np.testing.assert_allclose(result.cameras.fx.flatten().numpy(), [12, 15, 12, 15])
        self.assertEqual(result.cameras.metadata["frame_index"].flatten().tolist(), [0, 0, 1, 1])
        self.assertEqual(result.metadata["joint_params_init"].tolist(), [0, np.float32(.3)])


if __name__ == "__main__":
    unittest.main()
