"""Dense endpoints preserve every foreground depth pixel and its calibrated projection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'samuele'))
sys.path.insert(0, str(Path.home() / 'Joint3DGS/Sapien/scripts'))
import mask_colmap_models as shared
from read_write_model import qvec2rotmat
from test_real_recording_conversion import make_source
import convert_real_recording as converter
import dense_real_colmap as dense


class DenseRealTests(unittest.TestCase):
    def make_scene(self, root):
        source = make_source(root)
        calibration_path = source / 'meta/camera_calibration.json'
        calibration = json.loads(calibration_path.read_text())
        pose = np.array(calibration['cameras']['cam_1']['T_base_color_optical'])
        pose[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
        calibration['cameras']['cam_1']['T_base_color_optical'] = pose.tolist()
        calibration_path.write_text(json.dumps(calibration))
        scene = root / 'scene'
        converter.main(['--input',str(source),'--output',str(scene),'--size','32','24'])
        for view in ['cam_000', 'cam_001']:
            for frame in [0, 2]:
                path = scene / f'mask/{view}/frame_{frame:06d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((24,32), dtype=np.uint8)
                mask[3:20, 3:26] = 255
                cv2.imwrite(str(path), mask)
        return scene

    def test_every_pixel_tracks_projection_ply_and_preserved_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            scene = self.make_scene(Path(temp))
            seed = {p:p.read_bytes() for p in scene.glob('colmap*/sparse_unmasked/0/*')}
            dense.apply_scene(scene, shared)
            for model, frame, basename in [('colmap',0,'points3D'), ('colmap_final',2,'points3D_final')]:
                cameras, images, points = shared.read_model(scene/model/'sparse/0')
                expected = 0
                for iid, image in images.items():
                    self.assertEqual(Path(image.name).stem, f'frame_{frame:06d}')
                    depth = np.load(scene/'depth_npy'/Path(image.name).with_suffix('.npy'))
                    mask = cv2.imread(str(scene/'mask'/image.name), cv2.IMREAD_UNCHANGED)
                    ys, xs = np.nonzero((mask > 0) & np.isfinite(depth) & (depth > 0))
                    expected += len(xs)
                    np.testing.assert_array_equal(image.xys, np.column_stack([xs,ys]))
                    camera = cameras[image.camera_id]
                    fx,fy,cx,cy = camera.params
                    for index, pid in enumerate(image.point3D_ids):
                        point = points[pid]
                        local = qvec2rotmat(image.qvec) @ point.xyz + image.tvec
                        np.testing.assert_allclose([fx*local[0]/local[2]+cx,fy*local[1]/local[2]+cy], image.xys[index], atol=1e-10)
                        self.assertAlmostEqual(local[2], float(depth[ys[index],xs[index]]))
                        self.assertEqual(list(point.image_ids), [iid])
                        self.assertEqual(list(point.point2D_idxs), [index])
                self.assertEqual(len(points), expected)
                self.assertGreater(expected, 500)
                self.assertEqual((scene/model/'sparse/0/pointcloud.ply').read_bytes(), (scene/'pointcloud'/(basename+'.ply')).read_bytes())
                self.assertTrue((scene/'pointcloud'/(basename+'_unmasked.ply')).is_file())
            for path, original in seed.items(): self.assertEqual(path.read_bytes(), original)
            metadata = json.loads((scene/'metadata/colmap_export.json').read_text())
            self.assertEqual(metadata['point_stride'], 1)
            self.assertIsNone(metadata['voxel_size_m'])
            self.assertNotIn('fusion', metadata)
            # Re-running uses the preserved seed, rather than accumulating masked points.
            first = (scene/'colmap/sparse/0/points3D.bin').read_bytes()
            dense.apply_scene(scene, shared)
            self.assertEqual(first, (scene/'colmap/sparse/0/points3D.bin').read_bytes())

    def test_invalid_final_mask_does_not_publish_initial_model_or_ply(self):
        with tempfile.TemporaryDirectory() as temp:
            scene = self.make_scene(Path(temp))
            old = (scene/'pointcloud/points3D.ply').read_bytes()
            (scene/'mask/cam_001/frame_000002.png').unlink()
            with self.assertRaises(FileNotFoundError): dense.apply_scene(scene, shared)
            self.assertFalse((scene/'colmap/sparse/0').exists())
            self.assertEqual(old, (scene/'pointcloud/points3D.ply').read_bytes())

    def test_fraction_is_per_foreground_view_deterministic_and_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            scene = self.make_scene(Path(temp))
            dense.apply_scene(scene, shared, .125)
            for model in ['colmap', 'colmap_final']:
                _, images, points = shared.read_model(scene/model/'sparse/0')
                self.assertEqual(len(points), 2 * int(391*.125))
                self.assertTrue(all(len(image.xys) == int(391*.125) for image in images.values()))
            before = (scene/'colmap/sparse/0/points3D.bin').read_bytes()
            dense.apply_scene(scene, shared)
            self.assertEqual(before, (scene/'colmap/sparse/0/points3D.bin').read_bytes())
            metadata = json.loads((scene/'metadata/real_conversion.json').read_text())
            self.assertEqual(metadata['point_fraction'], .125)
            dense.apply_scene(scene, shared, 1)
            self.assertEqual(len(shared.read_model(scene/'colmap/sparse/0')[2]), 782)

    def test_invalid_fraction_does_not_modify_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            scene = self.make_scene(Path(temp))
            old = (scene/'pointcloud/points3D.ply').read_bytes()
            import argparse
            for fraction in [0, -1, 1.1, float('nan'), float('inf')]:
                with self.assertRaises(argparse.ArgumentTypeError): dense.apply_scene(scene, shared, fraction)
            self.assertEqual(old, (scene/'pointcloud/points3D.ply').read_bytes())


if __name__ == '__main__': unittest.main()
