import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'samuele'))
import export_real_robot_poses as exporter


class RobotPoseExportTests(unittest.TestCase):
    def fixture(self):
        info=dict(state_names=list(exporter.POSE_FIELDS)+['hand'])
        rows=[dict(frame_index=i,episode_index=7,timestamp=.02*i,
              **{'observation.state':[1+i,2,3,0,0,np.pi/2,999.]}) for i in range(3)]
        meta=dict(episode=7,source_frames=[0,2],output_frames=[0,1],coordinate_frame='base_link')
        return rows,info,meta

    def test_rotation_aware_tcp_offset_and_rgbd_source_frame_alignment(self):
        rows,info,meta=self.fixture();offset=np.eye(4);offset[0,3]=.2
        full,aligned=exporter.build_arrays(rows[::-1],info,meta,'ee',offset)
        np.testing.assert_allclose(full['tcp_position_m'][0],[1,2.2,3])
        np.testing.assert_allclose(full['ee_position_m'][0],[1,2,3])
        np.testing.assert_array_equal(full['frame_index'],[0,1,2])
        np.testing.assert_array_equal(aligned['frame_index'],[0,1])
        np.testing.assert_array_equal(aligned['source_frame_index'],[0,2])
        np.testing.assert_allclose(aligned['timestamp_s'],[0,.04])
        np.testing.assert_allclose(aligned['ee_position_m'][:,0],[1,3])

    def test_recorded_tcp_inverts_full_rigid_transform(self):
        rows,info,meta=self.fixture()
        calibration=np.eye(4);calibration[:3,:3]=Rotation.from_euler('x',90,degrees=True).as_matrix();calibration[:3,3]=[.1,.2,.3]
        full,_=exporter.build_arrays(rows,info,meta,'tcp',calibration)
        np.testing.assert_allclose(full['ee_transform']@calibration,full['tcp_transform'],atol=1e-12)

    def test_unknown_reference_does_not_fabricate_ee_or_tcp(self):
        rows,info,meta=self.fixture()
        full,_=exporter.build_arrays(rows,info,meta)
        self.assertIn('recorded_position_m',full)
        self.assertNotIn('ee_position_m',full);self.assertNotIn('tcp_position_m',full)
        with self.assertRaisesRegex(ValueError,'Declare --recorded-frame'):
            exporter.build_arrays(rows,info,meta,'unknown',np.eye(4))

    def test_invalid_and_missing_frames_are_errors(self):
        rows,info,meta=self.fixture()
        with self.assertRaisesRegex(ValueError,'unique nonnegative'):exporter.build_arrays(rows+[rows[0]],info,meta)
        meta['source_frames']=[0,9]
        with self.assertRaisesRegex(ValueError,'absent from Parquet'):exporter.build_arrays(rows,info,meta)
        meta['source_frames']=[0,2];meta['output_frames']=[0,1.5]
        with self.assertRaisesRegex(ValueError,'unique nonnegative'):exporter.build_arrays(rows,info,meta)

    def test_per_recording_calibration_and_episode_override(self):
        transform=np.eye(4);transform[2,3]=.2
        other=np.eye(4);other[2,3]=.03
        mapping=dict(defaults=dict(recorded_frame='ee'),datasets={
            'final_black_laptop':dict(T_ee_tcp=transform.tolist()),
            'final_black_laptop_episode_000007':dict(T_ee_tcp=other.tolist())})
        frame,matrix=exporter.calibration_for_scene(Path('final_black_laptop_episode_000000'),mapping,'unknown')
        self.assertEqual(frame,'ee');np.testing.assert_array_equal(matrix,transform)
        frame,matrix=exporter.calibration_for_scene(Path('final_black_laptop_episode_000007'),mapping,'unknown')
        np.testing.assert_array_equal(matrix,other)
        _,matrix=exporter.calibration_for_scene(Path('final_matchbox_episode_000000'),mapping,'unknown')
        self.assertIsNone(matrix)

    def test_confirmed_tool_calibration_uses_local_z_and_exact_scene_exceptions(self):
        config=Path(__file__).resolve().parents[1]/'configs/real_robot_pose_calibration.json'
        mapping=json.loads(config.read_text())
        rows,info,meta=self.fixture()
        for row in rows:row['observation.state'][3:6]=[0,np.pi/2,0]
        for episode in range(4):
            name=f'white_drawer_20261021_161154_episode_{episode:06d}'
            frame,matrix=exporter.calibration_for_scene(name,mapping,'unknown')
            offset=.03487365 if episode<3 else .1876174
            self.assertEqual(frame,'ee')
            self.assertEqual(matrix[2,3],offset)
            full,_=exporter.build_arrays(rows,info,meta,frame,matrix)
            delta=full['tcp_position_m']-full['ee_position_m']
            np.testing.assert_allclose(delta,np.tile([offset,0,0],(3,1)),atol=1e-12)
            np.testing.assert_allclose(full['tcp_transform'][:,:3,:3],full['ee_transform'][:,:3,:3])
        for name in ['final_robot_dog_episode_000000','final_black_laptop_episode_000000']:
            frame,matrix=exporter.calibration_for_scene(name,mapping,'unknown')
            self.assertEqual(frame,'ee');self.assertEqual(matrix[2,3],.1876174)

    def test_npz_export_no_pickle_and_missing_calibration_is_explicit(self):
        rows,info,meta=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            scene=Path(tmp)/'scene';raw=scene/'metadata/raw';raw.mkdir(parents=True)
            (raw/'info.json').write_text(json.dumps(info));(raw/'episode_000007.parquet').write_bytes(b'fixture')
            (scene/'metadata/real_conversion.json').write_text(json.dumps(meta))
            with patch.object(exporter,'read_parquet_rows',return_value=rows):
                report=exporter.export_scene(scene,Path(sys.executable),'ee',None)
                self.assertTrue(report['ee_available']);self.assertFalse(report['tcp_available'])
                output=scene/'metadata/robot_poses'
                with np.load(output/'poses_rgbd.npz',allow_pickle=False) as data:
                    self.assertEqual(data['ee_position_m'].shape,(2,3));self.assertNotIn('tcp_position_m',data.files)
                with self.assertRaises(FileExistsError):exporter.export_scene(scene,Path(sys.executable),'ee',None)
                before=(output/'poses_rgbd.npz').read_bytes()
                exporter.export_scene(scene,Path(sys.executable),'ee',None,True,True)
                self.assertEqual(before,(output/'poses_rgbd.npz').read_bytes())


if __name__=='__main__':unittest.main()
