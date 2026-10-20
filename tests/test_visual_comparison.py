import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from evaluate.visualize_local_comparison import (
    comparison_annotations, group_views, main, project_axis, raw_reference_image,
)


class VisualComparisonTest(unittest.TestCase):
    def prediction_fixture(self, root):
        scene, dataset, results = [root / name for name in ("scene", "dataset", "results")]
        sparse = scene / "colmap/sparse/0"
        sparse.mkdir(parents=True)
        (sparse / "images.txt").write_text(
            "1 1 0 0 0 0 0 0 1 cam_000/frame_000000.png\n\n"
            "2 1 0 0 0 -1 0 0 2 cam_001/frame_000000.png\n\n")
        dataset.mkdir()
        (results / "eval_results").mkdir(parents=True)
        rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
        translation = np.array([.2, -.4, .8])
        views, cameras = [], []
        for t in (0, 1):
            for c in (0, 1):
                stem = f"{len(views):05d}"
                view = dict(file_path=f"images/{stem}.png", camera_name=f"cam_{c:03d}",
                            time_index=t, source_stem=f"frame_{t*8:06d}",
                            fl_x=10, fl_y=10, cx=8, cy=6, w=16, h=12)
                views.append(view)
                pose = np.column_stack((rotation.T, rotation.T @ (np.array([c, 0, 0]) - translation)))
                cameras.append(dict(file_path=view["file_path"], transform=pose.tolist()))
                folder = scene / "rgb" / view["camera_name"]
                folder.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (16, 12), (70, 80, 90))
                image.save(folder / f"{view['source_stem']}.png")
                image.save(results / "eval_results" / f"rgb_{stem}.png")
        manifest = dict(source_scene=str(scene), cameras=["cam_000", "cam_001"], frames=views)
        (dataset / "multiview.json").write_text(json.dumps(manifest))
        (results / "transforms_train.json").write_text(json.dumps(cameras))
        articulation = dict(joint_type="revolute", joint_axis_world=(rotation.T @ [0, 1, 0]).tolist(),
                            joint_origin_world=(rotation.T @ (np.array([0, 0, 4]) - translation)).tolist(),
                            frames=[dict(frame_index=t, angle_rad=1 + .2*t) for t in (0, 1)])
        (results / "articulation.json").write_text(json.dumps(articulation))
        return scene, dataset, results, manifest

    def test_prediction_without_gt_aligns_axis_origin_and_relative_states(self):
        with tempfile.TemporaryDirectory() as temp:
            scene, dataset, results, manifest = self.prediction_fixture(Path(temp))
            report, states = comparison_annotations(scene, dataset, results, manifest)
            self.assertFalse(report["gt_available"])
            self.assertNotIn("joint_axis_error_deg", report)
            self.assertFalse((results / "joint_axis_error.json").exists())
            np.testing.assert_allclose(report["predicted_axis"], [0, 1, 0], atol=1e-7)
            np.testing.assert_allclose(report["predicted_origin"], [0, 0, 4], atol=1e-7)
            self.assertAlmostEqual(report["camera_position_max_error_m"], 0)
            self.assertAlmostEqual(states[0]["predicted_relative_state"], 0)
            self.assertAlmostEqual(states[1]["predicted_relative_state"], .2)
            self.assertEqual(states[1]["source_frame_index"], 8)

    def test_visual_debug_without_gt_or_masks_writes_estimated_overlays(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene, dataset, results, _ = self.prediction_fixture(root)
            out = root / "debug"
            with patch.object(sys, "argv", ["visualize", "--scene", str(scene), "--dataset", str(dataset),
                                            "--results", str(results), "--output-dir", str(out)]), \
                 patch("evaluate.visualize_local_comparison.subprocess.run") as encode:
                main()
            info = json.loads((out / "comparison_info.json").read_text())
            self.assertFalse(info["gt_available"])
            self.assertEqual(info["video"], "motion_axis_debug.mp4")
            self.assertEqual(info["timestamps"], 2)
            self.assertEqual(encode.call_count, 2)
            self.assertIn(str(out / "motion_axis_debug.mp4"), encode.call_args_list[0].args[0])
            image = np.asarray(Image.open(out / "comparison_00000.png"))
            # The predicted red axis appears on both the recorded and rendered panels.
            for panel in (image[88:328, :320], image[88:328, 320:640]):
                self.assertTrue(np.any((panel[:, :, 0] > 180) & (panel[:, :, 1] < 70) & (panel[:, :, 2] < 70)))

    def test_gt_axis_without_gt_states_still_allows_visual_debug(self):
        with tempfile.TemporaryDirectory() as temp:
            scene, dataset, results, manifest = self.prediction_fixture(Path(temp))
            (scene / "pointcloud").mkdir()
            (scene / "pointcloud/joint_metadata.json").write_text(json.dumps(dict(
                coordinate_frame="render_world", axis_vector=[0, 1, 0], point_of_application=[0, 0, 4])))
            report, states = comparison_annotations(scene, dataset, results, manifest)
            self.assertTrue(report["gt_available"])
            self.assertAlmostEqual(report["joint_axis_error_deg"], 0)
            self.assertNotIn("gt_relative_state", states[0])
            self.assertEqual(len(states), 2)

    def test_missing_duplicate_or_nonfinite_joint_states_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            scene, dataset, results, manifest = self.prediction_fixture(Path(temp))
            path = results / "articulation.json"
            original = json.loads(path.read_text())
            for frames in ([original["frames"][0]], original["frames"] + [original["frames"][0]],
                           [original["frames"][0], dict(frame_index=1, angle_rad=float("nan"))]):
                with self.subTest(frames=frames):
                    path.write_text(json.dumps(dict(original, frames=frames)))
                    with self.assertRaisesRegex(ValueError, "exactly once and be finite"):
                        comparison_annotations(scene, dataset, results, manifest)

    def test_projection_uses_calibrated_camera_and_rejects_behind_camera(self):
        pose = np.eye(4)
        pose[:3, 3] = [1, 2, 3]
        k = np.array([[100, 0, 50], [0, 120, 60], [0, 0, 1]])
        pixels = project_axis([1, 0, 0], [1, 2, 5], pose, k, 1)
        np.testing.assert_allclose(pixels, [[25, 60], [75, 60]])
        self.assertIsNone(project_axis([1, 0, 0], [1, 2, 2], pose, k, 1))

    def test_gt_background_compositing_preserves_foreground_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            scene = Path(temp)
            for folder in ("rgb", "mask"):
                (scene / folder / "cam_000").mkdir(parents=True)
            rgb = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
            mask = np.array([[255, 0], [255, 0]], dtype=np.uint8)
            Image.fromarray(rgb).save(scene / "rgb/cam_000/frame_000008.png")
            Image.fromarray(mask).save(scene / "mask/cam_000/frame_000008.png")
            image = np.asarray(raw_reference_image(scene, dict(camera_name="cam_000", source_stem="frame_000008")))
            np.testing.assert_array_equal(image[:, 0], rgb[:, 0])
            self.assertTrue(np.all(image[:, 1] == 255))

    def test_missing_or_duplicate_camera_observations_are_rejected(self):
        manifest = dict(cameras=["cam_000", "cam_001"], frames=[
            dict(time_index=t, camera_name=c) for t in (0, 1) for c in ("cam_000", "cam_001")])
        cameras, groups = group_views(manifest)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(cameras), 2)
        manifest["frames"].pop()
        with self.assertRaisesRegex(ValueError, "all cameras"):
            group_views(manifest)
        manifest["frames"].append(manifest["frames"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            group_views(manifest)


if __name__ == "__main__":
    unittest.main()
