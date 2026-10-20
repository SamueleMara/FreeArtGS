"""Geometry and file-format checks for the real RGB-D recording converter."""
import importlib.util
import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "samuele"))
import convert_real_recording as converter


def make_source(root):
    source = root / "raw"
    (source / "meta").mkdir(parents=True)
    cameras = {}
    for index in range(2):
        name = f"cam_{index}"
        video_dir = source / f"videos/chunk-000/observation.images.{name}"
        depth_dir = source / f"depth/chunk-000/observation.depth.{name}/episode_000000"
        video_dir.mkdir(parents=True); depth_dir.mkdir(parents=True)
        writer = cv2.VideoWriter(str(video_dir / "episode_000000.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25., (32,24))
        assert writer.isOpened()
        for frame in range(5):
            writer.write(np.full((24,32,3), frame * 40, dtype=np.uint8))
            if frame % 2 == 0:
                cv2.imwrite(str(depth_dir / f"frame_{frame:06d}.png"), np.full((24,32), 1000, dtype=np.uint16))
        writer.release()
        intrinsic = dict(width=32, height=24, K=[24,0,16,0,24,12,0,0,1], D=[0]*8, distortion_model="plumb_bob")
        pose = np.eye(4); pose[0,3] = index
        cameras[name] = dict(color=intrinsic, depth=dict(intrinsic, units="uint16_mm"),
                             T_color_depth=np.eye(4).tolist(), T_base_color_optical=pose.tolist())
    (source / "meta/camera_calibration.json").write_text(json.dumps(dict(cameras=cameras, base_frame="base_link", status="CANDIDATE_NOT_VALIDATED",
        extrinsic_convention="T_base_color_optical; p_base = T_base_color_optical * p_color_optical")))
    (source / "meta/info.json").write_text(json.dumps(dict(fps=25)))
    return source


class RealRecordingTests(unittest.TestCase):
    def test_sparse_depth_indices_select_correct_rgb_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = make_source(Path(temporary))
            video = source / "videos/chunk-000/observation.images.cam_0/episode_000000.mp4"
            frames = list(converter.selected_rgb(video, [0,2,4]))
            self.assertEqual([index for index, _ in frames], [0,2,4])
            for index, image in frames:
                self.assertAlmostEqual(float(image.mean()), index * 40, delta=5)

    def test_metric_depth_and_nearest_surface_z_buffer(self):
        rays = np.array([[0,0,1],[0,0,1],[1,0,1]], dtype=float)
        raw = np.array([[2000,1000,0]], dtype=np.uint16)
        depth = converter.register_depth(raw, rays, np.ones(3,dtype=bool), np.eye(4), np.eye(3), (3,2))
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(depth[0,0], 1)
        self.assertEqual(np.count_nonzero(depth), 1)

    def test_registration_applies_color_depth_translation_and_color_z(self):
        pose = np.eye(4); pose[:3,3] = [1,0,1]
        k = np.array([[4,0,2],[0,4,2],[0,0,1]], dtype=float)
        result = converter.register_depth(np.array([[1000]], dtype=np.uint16), np.array([[0,0,1.]]), np.array([True]), pose, k, (8,6))
        self.assertEqual(result[2,4], 2.)

    def test_rational_depth_distortion_is_inverted(self):
        camera = dict(width=16,height=12,K=[20,0,8,0,20,6,0,0,1], D=[.2,.03,.001,-.002,.01,.1,.01,.001],distortion_model="plumb_bob")
        rays, accepted = converter.depth_rays(camera)
        self.assertTrue(accepted.all())
        projected,_ = cv2.projectPoints(rays, np.zeros(3),np.zeros(3),np.array(camera["K"]).reshape(3,3).astype(float),np.array(camera["D"]))
        yy,xx=np.mgrid[:12,:16]
        np.testing.assert_allclose(projected.reshape(12,16,2), np.stack([xx,yy],axis=-1), atol=1e-6)

    def test_complete_conversion_and_colmap_tracks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=make_source(root); output=root/"converted"
            for p in source.glob("depth/**/frame_000004.png"):
                cv2.imwrite(str(p), np.full((24,32),2000,dtype=np.uint16))
            before={p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            converter.main(["--input",str(source),"--output",str(output),"--size","32","24","--point-stride","8"])
            self.assertFalse((output/"mask").exists())
            self.assertFalse((output/"conversion_in_progress.json").exists())
            for camera in ["cam_000","cam_001"]:
                self.assertEqual(sorted(p.stem for p in (output/"rgb"/camera).glob("*.png")), ["frame_000000","frame_000001","frame_000002"])
                depth=np.load(output/f"depth_npy/{camera}/frame_000002.npy")
                self.assertEqual(depth[12,16],2.)
                self.assertEqual(cv2.imread(str(output/f"depth_png/{camera}/frame_000002.png"),-1)[12,16],2000)
                for state in ["start","end"]:
                    self.assertTrue((output/f"pointcloud/views/{state}/{camera}.ply").is_file())
            metadata=json.loads((output/"metadata/real_conversion.json").read_text())
            self.assertEqual(metadata["calibration_status"],"CANDIDATE_NOT_VALIDATED")
            self.assertFalse(metadata["joint_gt_available"])
            self.assertEqual(metadata["source_frames"],[0,2,4])
            self.assertEqual(metadata["output_frames"],[0,1,2])
            self.assertEqual((output/"metadata/frame_timestamps.csv").read_text().splitlines(),
                             ["frame_index,source_frame_index,timestamp_s", "0,0,0.0", "1,2,0.08", "2,4,0.16"])
            for camera in ["cam_000","cam_001"]:
                for number, source_frame in enumerate([0,2,4]):
                    image=cv2.imread(str(output/f"rgb/{camera}/frame_{number:06d}.png"))
                    self.assertAlmostEqual(float(image[12,16].mean()), source_frame*40, delta=5)
            # Validate the binary model independently of the writer, including track references.
            self.assertEqual(len(list((output/"colmap/images").glob("*/*.png"))),2)
            self.assertEqual(json.loads((output/"metadata/colmap_export.json").read_text())["num_images"],2)
            sparse=output/"colmap/sparse_unmasked/0"
            with (sparse/"images.bin").open("rb") as stream:
                self.assertEqual(struct.unpack("<Q",stream.read(8))[0],2)
                observations={}
                for _ in range(2):
                    values=struct.unpack("<idddddddi",stream.read(64)); image_id=values[0]
                    quaternion=values[1:5]; translation=values[5:8]; camera_id=values[8]
                    np.testing.assert_allclose(Rotation.from_quat([*quaternion[1:],quaternion[0]]).as_matrix(),np.eye(3),atol=1e-7)
                    np.testing.assert_allclose(translation,[-(camera_id-1),0,0])
                    name=bytearray()
                    while (char:=stream.read(1)) != b"\0":
                        self.assertTrue(char); name.extend(char)
                    self.assertTrue((output/"colmap/images"/name.decode()).is_file())
                    self.assertEqual(Path(name.decode()).name,"frame_000000.png")
                    count=struct.unpack("<Q",stream.read(8))[0]
                    observations[image_id]=[struct.unpack("<ddq",stream.read(24)) for _ in range(count)]
                self.assertFalse(stream.read(1))
            with (sparse/"points3D.bin").open("rb") as stream:
                count=struct.unpack("<Q",stream.read(8))[0]
                self.assertGreater(count,0)
                for _ in range(count):
                    values=struct.unpack("<QdddBBBdQ",stream.read(51))
                    pid=values[0]
                    self.assertAlmostEqual(values[3],1.)
                    for _ in range(values[-1]):
                        image_id,point_index=struct.unpack("<ii",stream.read(8))
                        self.assertEqual(observations[image_id][point_index][2],pid)
                self.assertFalse(stream.read(1))
            final=output/"colmap_final/sparse_unmasked/0"
            summary=json.loads((output/"metadata/colmap_export_final.json").read_text())
            self.assertEqual(summary["endpoint_frame"],2)
            self.assertEqual(summary["num_images"],2)
            self.assertTrue((output/"pointcloud/points3D_final.ply").is_file())
            final_names=[line.split()[-1] for line in (final/"images.txt").read_text().splitlines() if 'frame_' in line]
            self.assertEqual(final_names,["cam_000/frame_000002.png","cam_001/frame_000002.png"])
            for name in final_names:
                self.assertEqual((output/"colmap_final/images"/name).read_bytes(),(output/"rgb"/name).read_bytes())
            with (final/"points3D.bin").open("rb") as stream:
                final_count=struct.unpack("<Q",stream.read(8))[0]
                self.assertGreater(final_count,0)
                for _ in range(final_count):
                    values=struct.unpack("<QdddBBBdQ",stream.read(51))
                    self.assertAlmostEqual(values[3],2.)
                    stream.read(values[-1]*8)
                self.assertFalse(stream.read(1))
            protected={p:p.read_bytes() for folder in ["rgb","depth_npy","depth_png","colmap"] for p in (output/folder).rglob("*") if p.is_file()}
            mask=output/"mask/cam_000/frame_000002.png"
            mask.parent.mkdir(parents=True);mask.write_bytes(b"separately-created-mask")
            protected[mask]=mask.read_bytes()
            shutil.rmtree(output/"colmap_final")
            converter.main(["--output",str(output),"--final-colmap-only","--point-stride","8"])
            for p,value in protected.items():self.assertEqual(p.read_bytes(),value)
            self.assertEqual(json.loads((output/"metadata/colmap_export_final.json").read_text())["endpoint_frame"],2)
            with self.assertRaises(FileExistsError):
                converter.main(["--output",str(output),"--final-colmap-only"])
            for p,value in before.items(): self.assertEqual(p.read_bytes(),value)
            with self.assertRaises(FileExistsError):
                converter.main(["--input",str(source),"--output",str(output)])

    def test_dry_run_creates_no_output_and_missing_depth_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=make_source(root);output=root/"absent"
            converter.main(["--input",str(source),"--output",str(output),"--dry-run"])
            self.assertFalse(output.exists())
            for p in source.glob("depth/**/*.png"):p.unlink()
            with self.assertRaises(ValueError):converter.main(["--input",str(source),"--output",str(output)])
            self.assertFalse(output.exists())

    def test_multiview_voxel_fusion_averages_geometry_color_and_builds_tracks(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            images=[];cameras=[]
            for cid,name,color in [(1,"cam_000",[10,0,0]),(2,"cam_001",[0,0,20])]:
                (output/"rgb"/name).mkdir(parents=True)
                (output/"depth_npy"/name).mkdir(parents=True)
                np.save(output/f"depth_npy/{name}/frame_000004.npy",np.ones((2,2),dtype=np.float32))
                cv2.imwrite(str(output/f"rgb/{name}/frame_000004.png"),np.full((2,2,3),color[::-1],dtype=np.uint8))
                cameras.append(dict(id=cid,name=name,pose_matrix_4x4=np.eye(4).tolist()))
                images.append(dict(id=cid,camera_id=cid,name=f"{name}/frame_000004.png",observations=[]))
            points,summary=converter.merge_endpoint_clouds(output,cameras,images,(2,2),np.eye(3),1,.005,"end")
            self.assertEqual(summary["source_points"],8)
            self.assertEqual(summary["merged_points"],4)
            self.assertEqual(summary["multi_camera_points"],4)
            for pid,xyz,color,error,tracks in points:
                self.assertEqual(color,[5,0,10])
                self.assertEqual(len(tracks),2)
                self.assertAlmostEqual(error,0)
                for image_id,idx in tracks:
                    self.assertEqual(images[image_id-1]["observations"][idx][2],pid)


if __name__ == "__main__":
    unittest.main()
