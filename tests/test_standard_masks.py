import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "samuele"))
import create_standard_masks as masker


class StandardMaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for camera in ["cam_000", "cam_001"]:
            (self.root/"rgb"/camera).mkdir(parents=True)
            for t in [0,2,4]:
                Image.fromarray(np.full((6,8,3),t+50,dtype=np.uint8)).save(self.root/f"rgb/{camera}/frame_{t:06d}.png")
        self.args = argparse.Namespace(input=self.root,prompt="red fridge",model_root=Path('/local/models'),box_thr=.25,text_thr=.25,
                                       max_frames=0,overwrite=False,dry_run=False,device='cuda')

    def tearDown(self):
        self.temporary.cleanup()

    def run_generation(self, segment):
        with patch.dict(os.environ), patch.object(tempfile, "tempdir", tempfile.tempdir):
            return masker.generate(self.args,segment)

    def test_prompt_binary_layout_resolution_and_sources(self):
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.root/"rgb").rglob('*.png')}
        seen=[]
        def segment(path,prompt):
            seen.append(prompt)
            mask=np.zeros((6,8),dtype=np.uint8);mask[2:5,1:7]=255
            return mask,dict(detections=1)
        report=self.run_generation(segment)
        self.assertEqual(seen,["red fridge"]*6)
        self.assertEqual(report['completed'],6)
        self.assertEqual(report['empty'],0)
        self.assertFalse((self.root/"masks").exists())
        for path in before:
            target=self.root/"mask"/path.relative_to(self.root/"rgb")
            mask=np.asarray(Image.open(target))
            self.assertEqual(mask.shape,(6,8))
            self.assertEqual(set(np.unique(mask)),{0,255})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before[path])

    def test_resume_skips_models_and_changed_prompt_requires_overwrite(self):
        def segment(path,prompt):return np.full((6,8),255,dtype=np.uint8),dict(detections=1)
        self.run_generation(segment)
        def forbidden(*args):raise AssertionError('Inference should be skipped')
        with patch.object(masker,'load_segmenter',side_effect=AssertionError('Should not load model')):
            report=self.run_generation(forbidden)
        self.assertEqual(report['skipped'],6)
        self.args.prompt='refrigerator'
        with self.assertRaisesRegex(ValueError,'settings/prompt changed'):self.run_generation(forbidden)
        self.args.overwrite=True
        self.assertEqual(self.run_generation(segment)['settings']['prompt'],'refrigerator')

    def test_changed_rgb_is_not_silently_reused(self):
        segment=lambda path,prompt:(np.full((6,8),255,dtype=np.uint8),dict(detections=1))
        self.run_generation(segment)
        Image.fromarray(np.zeros((6,8,3),dtype=np.uint8)).save(self.root/'rgb/cam_000/frame_000000.png')
        with self.assertRaisesRegex(ValueError,'RGB changed'):self.run_generation(segment)

    def test_empty_masks_have_files_and_explicit_status(self):
        report=self.run_generation(lambda path,prompt:(np.zeros((6,8),dtype=np.uint8),dict(detections=0)))
        self.assertEqual(report['empty'],6)
        self.assertTrue(all(row['status']=='empty' for row in report['frames'].values()))
        self.assertEqual(len(list((self.root/'mask').rglob('*.png'))),6)

    def test_dry_run_has_no_writes_and_limits_each_camera(self):
        self.args.max_frames=1;self.args.dry_run=True
        report=self.run_generation(None)
        self.assertEqual(report['requested'],2)
        self.assertFalse((self.root/'mask').exists())
        self.assertFalse((self.root/'.runtime').exists())

    def test_complete_mask_generation_publishes_colmap_pairs(self):
        for model in ["colmap", "colmap_final"]:
            (self.root/model).mkdir()
        with patch('colmap_masking.apply_scene_masks') as publish:
            self.run_generation(lambda path,prompt:(np.full((6,8),255,dtype=np.uint8),dict(detections=1)))
            publish.assert_called_once_with(self.root)
        self.args.max_frames=1
        with patch('colmap_masking.apply_scene_masks') as publish:
            self.run_generation(None)
            publish.assert_not_called()

    def test_point_fraction_is_forwarded_without_changing_mask_resume_settings(self):
        for model in ['colmap', 'colmap_final']: (self.root/model).mkdir()
        segment = lambda path,prompt:(np.full((6,8),255,dtype=np.uint8),dict(detections=1))
        with patch('colmap_masking.apply_scene_masks') as publish:
            self.run_generation(segment)
            self.args.point_fraction = .125
            self.run_generation(lambda *args: self.fail('Changing point sampling must reuse masks'))
            publish.assert_called_with(self.root, fraction=.125)

    def test_wrong_size_mask_rejected(self):
        with self.assertRaisesRegex(ValueError,'original RGB dimensions'):
            self.run_generation(lambda path,prompt:(np.zeros((3,4),dtype=np.uint8),{}))
        self.assertFalse(list((self.root/'mask').rglob('*.png')))


if __name__ == '__main__':unittest.main()
