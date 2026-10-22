import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'samuele'))
from leica_mask_guidance import LeicaGuidance, read_xyz, large_qr_occluder
import cv2


class LeicaGuidanceTests(unittest.TestCase):
    def test_large_calibration_board_is_an_exclusion_not_foreground(self):
        encoded=cv2.QRCodeEncoder_create().encode('calibration board')
        canvas=np.full((480,640,3),255,np.uint8)
        canvas[70:410,150:490]=cv2.resize(cv2.cvtColor(encoded,cv2.COLOR_GRAY2RGB),(340,340),interpolation=cv2.INTER_NEAREST)
        self.assertIsNotNone(large_qr_occluder(canvas))
        self.assertIsNone(large_qr_occluder(np.full((480,640,3),255,np.uint8)))

    def test_binary_and_ascii_cloud_and_detection_disambiguation(self):
        with tempfile.TemporaryDirectory() as temp:
            scene=Path(temp);cloud=scene/'part.ply'
            xy=np.array([[x,y,2.] for x in np.linspace(-.3,.3,10) for y in np.linspace(-.3,.3,10)])
            header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xy)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n'
            cloud.write_bytes(header.encode()+xy.astype('<f8').tobytes())
            np.testing.assert_allclose(read_xyz(cloud),xy)
            (scene/'metadata/poses').mkdir(parents=True)
            (scene/'metadata/leica_moving_part.json').write_text(json.dumps(dict(path='part.ply',coordinate_frame='hoi_leica_world',units='meters',object_index=1)))
            meta=dict(coordinate_frame='hoi_leica_world',width=100,height=100,intrinsics_matrix_3x3=[[100,0,50],[0,100,50],[0,0,1]],frames=[dict(frame_index=22,world_to_camera_matrix_4x4=np.eye(4).tolist())])
            (scene/'metadata/poses/cam_002.json').write_text(json.dumps(meta))
            image=scene/'rgb/cam_002/frame_000022.png'
            guide=LeicaGuidance();index,details=guide.select(image,np.array([[0,0,20,100],[30,30,70,70]]))
            self.assertEqual(index,1);self.assertEqual(details['selection_status'],'selected')
            index,details=guide.select(image,np.array([[0,0,20,100]]))
            self.assertIsNone(index);self.assertEqual(details['selection_status'],'leica_box_fallback')
            self.assertTrue(details['sam_positive_points'])
            np.savetxt(cloud,xy,header=header.replace('binary_little_endian','ascii').strip(),comments='')
            np.testing.assert_allclose(read_xyz(cloud),xy)


if __name__=='__main__':unittest.main()
