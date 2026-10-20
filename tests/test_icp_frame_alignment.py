import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from icp.color_icp_gpu import _to_tensor_pcd, build_pose_graph, load_frame_point_clouds, write_transforms_json


class IcpFrameAlignmentTest(unittest.TestCase):
    def test_missing_cloud_keeps_slot_and_export_keeps_rgb_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "point_clouds_0"
            folder.mkdir()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector([[0, 0, 1]])
            o3d.io.write_point_cloud(str(folder / "00007.ply"), cloud)
            o3d.io.write_point_cloud(str(folder / "00009.ply"), cloud)
            stems = ["00007", "00008", "00009"]
            with contextlib.redirect_stdout(io.StringIO()):
                clouds = load_frame_point_clouds(root, 0, stems)
                write_transforms_json(root, 0, [np.eye(4)] * 3, np.eye(3), 6, 8,
                                      valid_mask=[True, False, True], frame_stems=stems)
            self.assertEqual([len(cloud.points) for cloud in clouds], [1, 0, 1])
            frames = json.loads((root / "transforms_0.json").read_text())["frames"]
            self.assertEqual([frame["file_path"] for frame in frames], ["images/00007.png", "images/00009.png"])

    def test_empty_parts_keep_finite_pose_slots(self):
        args = SimpleNamespace(voxel_size=.02, global_opt=False, loop_interval=0)
        clouds = [o3d.geometry.PointCloud() for _ in range(3)]
        tensors = [_to_tensor_pcd(cloud, o3d.core.Device("CPU:0")) for cloud in clouds]
        _, poses, valid = build_pose_graph(args, clouds, tensor_point_clouds=tensors)
        self.assertEqual(np.asarray(poses).shape, (3, 4, 4))
        self.assertTrue(np.isfinite(poses).all())
        self.assertEqual(valid, [False, False, False])


if __name__ == "__main__":
    unittest.main()
