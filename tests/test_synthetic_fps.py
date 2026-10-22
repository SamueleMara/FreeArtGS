import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'samuele'))
import add_synthetic_fps as updater


class SyntheticFPSTests(unittest.TestCase):
    def scene(self, root, name, fps=None):
        scene = root/name
        (scene/'metadata').mkdir(parents=True)
        (scene/'metadata/job_config_resolved.yaml').write_text('num_frames: 3\nfps: null\n')
        (scene/'metadata/cameras.json').write_text(json.dumps({'cameras':[{'name':'cam_000'}]}))
        if fps:
            (scene/'video_rgb').mkdir()
            video = cv2.VideoWriter(str(scene/'video_rgb/cam_000.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(16,16))
            self.assertTrue(video.isOpened())
            for _ in range(3): video.write(np.zeros((16,16,3),dtype=np.uint8))
            video.release()
        return scene

    def test_video_and_estimate_provenance_and_safe_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_scene = self.scene(root,'preparation_8cams/profile/object',59)
            estimate = self.scene(root,'preparation_100cams/object')
            self.scene(root,'real/ignored',25)
            self.assertEqual(len(updater.discover(root)),2)
            updater.main(['--root',str(root)])
            for scene, source in [(video_scene,'video_metadata'),(estimate,'provided_video_estimate')]:
                data = json.loads((scene/'metadata/timing.json').read_text())
                self.assertEqual(data['fps'],59.)
                self.assertEqual(data['fps_source'],source)
                config = yaml.safe_load((scene/'metadata/job_config_resolved.yaml').read_text())
                self.assertIsNone(config['fps'])
                self.assertEqual(config['num_frames'],3)
                self.assertEqual(config['fps_resolved'],59.)
                self.assertEqual(json.loads((scene/'metadata/cameras.json').read_text())['fps'],59.)
            before = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            updater.main(['--root',str(root)])
            self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))

    def test_mismatched_video_and_dry_run_do_not_change_any_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.scene(root,'a',59)
            b=self.scene(root,'b',30)
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaisesRegex(ValueError,'30.0 FPS'):
                updater.main(['--root',str(root)])
            self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))
            with patch.object(updater,'prepare',return_value=({},59,1)):
                updater.main(['--root',str(b),'--dry-run'])
            self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))


if __name__=='__main__': unittest.main()
