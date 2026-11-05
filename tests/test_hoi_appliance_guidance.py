import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,'/home/samuelemara/Joint3DGS/Sapien/scripts')
from create_hoi_appliance_guidance import create_guidance


class ApplianceGuidanceTests(unittest.TestCase):
    def test_user_groups_and_negative_points(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw';scene=root/'scene';(scene/'metadata').mkdir(parents=True)
            base=raw/'fridge_1/leica/001/instance_annotations_3d/points';base.mkdir(parents=True)
            for index in range(1,5):
                points=np.array([[index,y,z] for y in [.1,.2,.3] for z in [.1,.2,.3]])
                header=f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\nend_header'
                np.savetxt(base/f'{index:03d}.ply',points,header=header,comments='')
            for index,positive,negative in [(1,[1,2],[3,4]),(4,[3,4],[1,2])]:
                (scene/'metadata/hoi_conversion.json').write_text(json.dumps(dict(location='fridge_1',object_index=index)))
                create_guidance(scene,raw)
                data=json.loads((scene/'metadata/leica_appliance_guidance.json').read_text())
                self.assertEqual(data['positive_instance_ids'],positive)
                self.assertEqual(data['negative_instance_ids'],negative)
                self.assertEqual(set(np.asarray(data['positive_world_xyz'])[:,0]),set(positive))
                self.assertEqual(set(np.asarray(data['negative_world_xyz'])[:,0]),set(negative))


if __name__=='__main__':unittest.main()
