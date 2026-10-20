"""Regression: colliding sparse capture IDs must migrate every modality/reference."""
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'samuele'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_real_recording as converter
import renumber_real_recording as repair
from test_real_recording_conversion import make_source


class RenumberTests(unittest.TestCase):
    def legacy(self, root):
        source = make_source(root)
        output = root / 'scene'
        converter.main(['--input', str(source), '--output', str(output), '--size', '32', '24'])
        for model in ['colmap', 'colmap_final']:
            (output / model / 'sparse_unmasked').rename(output / model / 'sparse')
        inverse = {0: 0, 1: 2, 2: 4}
        for camera in ['cam_000', 'cam_001']:
            (output / 'mask' / camera).mkdir(parents=True)
            for frame in range(3):
                (output / f'mask/{camera}/frame_{frame:06d}.png').write_bytes(b'mask-' + bytes([frame]))
        manifest = {'settings': {'prompt': 'test'}, 'frames': {
            f'{camera}/frame_{frame:06d}.png': {'rgb_sha256': repair.digest(output / f'rgb/{camera}/frame_{frame:06d}.png')}
            for camera in ['cam_000', 'cam_001'] for frame in range(3)}}
        converter.write_json(output / 'metadata/mask_generation.json', manifest)
        expected = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('frame_*') if p.suffix in ['.png', '.npy']}
        # Manufacture the previous converter's sparse numbering, including collisions.
        staged = []
        for i, p in enumerate(sorted(output.rglob('frame_*'))):
            if p.suffix in ['.png', '.npy']:
                final = p.with_name(repair.rename_string(p.name, inverse))
                temp = root / f'temp-{i}'
                p.rename(temp)
                staged.append((temp, final))
        for temp, final in staged:
            temp.rename(final)
        for p in output.rglob('*'):
            if p.suffix == '.json':
                data = repair.rename_json(json.loads(p.read_text()), inverse)
                if p.name == 'real_conversion.json':
                    data.pop('output_frames'); data.pop('frame_naming')
                converter.write_json(p, data)
            elif p.name == 'images.txt':
                p.write_text(repair.rename_string(p.read_text(), inverse))
            elif p.name == 'images.bin':
                p.write_bytes(repair.rename_images_binary(p.read_bytes(), inverse)[0])
        (output / 'metadata/frame_timestamps.csv').write_text('frame_index,timestamp_s\n0,0.0\n2,0.08\n4,0.16\n')
        return output, expected

    def test_repair_preserves_content_tracks_provenance_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, expected = self.legacy(Path(temporary))
            protected = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*')
                         if p.is_file() and (p.name in ['cameras.bin', 'points3D.bin'] or p.suffix == '.ply')}
            before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
            repair.renumber(output, dry_run=True)
            self.assertEqual(before, {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()})
            original_rename = Path.rename
            installs = 0
            def interrupt(path, destination):
                nonlocal installs
                if '/frame_renumbering_backup/files/' in str(path):
                    installs += 1
                    if installs == 3:
                        raise OSError('simulated interruption during install')
                return original_rename(path, destination)
            with patch.object(Path, 'rename', interrupt):
                with self.assertRaisesRegex(OSError, 'simulated interruption'):
                    repair.renumber(output)
            repair.renumber(output)
            repair.renumber(output)  # Idempotent, including an already-complete journal.
            for name, data in (expected | protected).items():
                self.assertEqual((output / name).read_bytes(), data, name)
            identity = {0: 0, 1: 1, 2: 2}
            for model in ['colmap', 'colmap_final']:
                sparse = output / model / 'sparse/0'
                _, names = repair.rename_images_binary((sparse / 'images.bin').read_bytes(), identity)
                text_names = [line.split()[-1] for line in (sparse / 'images.txt').read_text().splitlines() if 'frame_' in line]
                self.assertEqual(names, text_names)
                for name in names:
                    self.assertTrue((output / model / 'images' / name).is_file())
            manifest = json.loads((output / 'metadata/mask_generation.json').read_text())
            for name, item in manifest['frames'].items():
                self.assertEqual(repair.digest(output / 'rgb' / name), item['rgb_sha256'])
                self.assertTrue((output / 'mask' / name).is_file())
            self.assertEqual(json.loads((output / 'metadata/colmap_export_final.json').read_text())['endpoint_frame'], 2)
            metadata = json.loads((output / 'metadata/real_conversion.json').read_text())
            self.assertEqual(metadata['source_frames'], [0, 2, 4])
            self.assertEqual(metadata['output_frames'], [0, 1, 2])
            rows = list(csv.DictReader(io.StringIO((output / 'metadata/frame_timestamps.csv').read_text())))
            self.assertEqual([r['source_frame_index'] for r in rows], ['0', '2', '4'])
            self.assertEqual([r['timestamp_s'] for r in rows], ['0.0', '0.08', '0.16'])

    def test_incomplete_modalities_and_malformed_binary_fail_before_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = self.legacy(Path(temporary))
            (output / 'depth_npy/cam_000/frame_000002.npy').unlink()
            with self.assertRaisesRegex(ValueError, 'missing or unexpected frames'):
                repair.renumber(output)
            self.assertFalse((output / repair.JOURNAL).exists())
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                repair.rename_images_binary(b'\x01', {0: 0})


if __name__ == '__main__':
    unittest.main()
