import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate.evaluate_local_joint_axis import evaluate_joint_axis, load_gt_axis, joint_motion_metrics
from preprocessing.prepare_multiview import calibration_poses


class JointAxisEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.scene, self.dataset, self.results = [root / name for name in ("scene", "dataset", "results")]
        sparse = self.scene / "colmap/sparse/0"
        sparse.mkdir(parents=True)
        q = np.sqrt(.5)
        (sparse / "images.txt").write_text(
            f"1 {q} 0 0 {q} 0 0 0 1 cam_000/frame_000000.png\n\n"
            "2 1 0 0 0 1 2 3 2 cam_001/frame_000000.png\n\n")
        (self.scene / "pointcloud").mkdir()
        (self.scene / "pointcloud/joint_metadata.json").write_text(json.dumps(
            dict(axis_vector=[0, 0, 2], coordinate_frame="render_world")))
        self.dataset.mkdir()
        frames = [dict(file_path=f"images/{i:05d}.png", camera_name=f"cam_{i % 2:03d}",
                       time_index=i // 2) for i in range(4)]
        (self.dataset / "multiview.json").write_text(json.dumps(
            dict(source_scene=str(self.scene), frames=frames)))
        self.results.mkdir()
        # A deliberately different prediction coordinate frame exposes missing frame alignment.
        self.alignment = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        calibrated = calibration_poses(self.scene)
        self.cameras = []
        for frame in frames:
            pose = np.eye(4)
            pose[:3, :3] = self.alignment.T @ calibrated[frame["camera_name"]][:3, :3]
            self.cameras.append(dict(file_path=frame["file_path"], transform=pose[:3].tolist()))
        (self.results / "transforms_train.json").write_text(json.dumps(self.cameras))

    def evaluate(self, render_axis):
        axis = self.alignment.T @ render_axis
        (self.results / "articulation.json").write_text(json.dumps(dict(joint_axis_world=axis.tolist())))
        return evaluate_joint_axis(self.scene, self.dataset, self.results)

    def test_sign_invariance_coordinate_alignment_and_degrees(self):
        self.assertAlmostEqual(self.evaluate(np.array([0., 0., -3.]))["joint_axis_error_deg"], 0)
        angle = np.deg2rad(10)
        report = self.evaluate(np.array([np.sin(angle), 0., np.cos(angle)]))
        self.assertAlmostEqual(report["joint_axis_error_deg"], 10)
        np.testing.assert_allclose(report["prediction_to_render_rotation"], self.alignment, atol=1e-6)
        self.assertTrue((self.results / "joint_axis_error.json").is_file())
        self.assertAlmostEqual(self.evaluate(np.array([1., 0., 0.]))["joint_axis_error_deg"], 90)

    def test_rejects_invalid_gt_and_inconsistent_calibration(self):
        path = self.scene / "pointcloud/joint_metadata.json"
        path.write_text(json.dumps(dict(axis_vector=[0, 0, 0], coordinate_frame="render_world")))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            load_gt_axis(self.scene)
        path.write_text(json.dumps(dict(axis_vector=[0, 0, 1], coordinate_frame="urdf")))
        with self.assertRaisesRegex(ValueError, "render_world"):
            load_gt_axis(self.scene)
        path.write_text(json.dumps(dict(axis_vector=[0, 0, 1], coordinate_frame="render_world")))
        self.cameras[1]["transform"] = np.eye(4)[:3].tolist()
        (self.results / "transforms_train.json").write_text(json.dumps(self.cameras))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.evaluate(np.array([0., 0., 1.]))

    def test_gt_overlay_uses_calibrated_rotation_and_translation(self):
        translation = np.array([.2, -.4, .8])
        calibrated = calibration_poses(self.scene)
        for i, camera in enumerate(self.cameras):
            pose = np.asarray(camera["transform"])
            pose[:3, 3] = self.alignment.T @ (calibrated[f"cam_{i % 2:03d}"][:3, 3] - translation)
            camera["transform"] = pose.tolist()
        (self.results / "transforms_train.json").write_text(json.dumps(self.cameras))
        gt_origin = np.array([1., 2., 3.])
        (self.scene / "pointcloud/joint_metadata.json").write_text(json.dumps(dict(
            axis_vector=[0, 0, 1], coordinate_frame="render_world", point_of_application=gt_origin.tolist())))
        predicted_origin = self.alignment.T @ (gt_origin - translation)
        (self.results / "articulation.json").write_text(json.dumps(dict(
            joint_axis_world=(self.alignment.T @ [0, 0, 1]).tolist(),
            joint_origin_world=predicted_origin.tolist())))
        report = evaluate_joint_axis(self.scene, self.dataset, self.results)
        np.testing.assert_allclose(report["prediction_to_render_translation"], translation)
        np.testing.assert_allclose(report["predicted_origin"], gt_origin)
        np.testing.assert_allclose(report["gt_origin_in_prediction_frame"], predicted_origin)

    def motion_fixture(self, gt_type="prismatic"):
        (self.scene / "metadata").mkdir()
        (self.scene / "metadata/frame_values.csv").write_text("frame_index,slider\n0,2\n8,2.4\n")
        (self.scene / "pointcloud/joint_metadata.json").write_text(json.dumps(dict(name="slider", type=gt_type)))
        manifest = dict(frames=[dict(time_index=t, source_stem=f"frame_{s:06d}")
                               for t, s in ((0, 0), (0, 0), (1, 8), (1, 8))])
        articulation = dict(joint_type="prismatic", frames=[dict(frame_index=0, distance=7),
                                                            dict(frame_index=1, distance=6.5)])
        return manifest, articulation

    def test_state_error_maps_source_frames_axis_sign_and_offset_without_fitting(self):
        manifest, articulation = self.motion_fixture()
        report = joint_motion_metrics(self.scene, manifest, articulation, -1)
        self.assertEqual(report["evaluated_timestamps"], 2)
        self.assertEqual(report["joint_state_unit"], "m")
        self.assertAlmostEqual(report["joint_state_mae"], .05)
        self.assertAlmostEqual(report["predicted_motion_range"], .5)
        self.assertAlmostEqual(report["gt_motion_range"], .4)
        self.assertEqual(report["joint_state_samples"][1]["source_frame_index"], 8)
        self.assertAlmostEqual(report["joint_state_samples"][1]["predicted_relative_state"], .5)
        self.assertAlmostEqual(report["joint_state_samples"][1]["gt_relative_state"], .4)

    def test_state_error_reports_type_mismatch(self):
        manifest, articulation = self.motion_fixture("continuous")
        report = joint_motion_metrics(self.scene, manifest, articulation, 1)
        self.assertFalse(report["joint_type_correct"])
        self.assertEqual(report["gt_joint_type"], "revolute")
        self.assertEqual(report["joint_state_evaluation"], "type_mismatch")
        self.assertNotIn("joint_state_mae", report)

    def test_state_error_rejects_missing_timestamps(self):
        manifest, articulation = self.motion_fixture()
        articulation["frames"].pop()
        with self.assertRaisesRegex(ValueError, "exactly once"):
            joint_motion_metrics(self.scene, manifest, articulation, 1)


if __name__ == "__main__":
    unittest.main()
