from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'samuele'))
import prepare_real_dataset_batch as batch


class RealBatchTests(unittest.TestCase):
    def named_source(self, root, name, episodes):
        source=root/name;(source/'meta').mkdir(parents=True)
        (source/'meta/episodes.jsonl').write_text('\n'.join(json.dumps(dict(episode_index=e)) for e in episodes))
        (source/'meta/tasks.jsonl').write_text('{"task": "incorrect old label"}')
        return source

    def test_invalid_later_episode_prevents_deletion_of_any_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'dataset_20261021_161154';source.mkdir();outputs=root/'out';outputs.mkdir()
            old=outputs/'white_drawer_20261021_161154_episode_000000';old.mkdir();(old/'keep').write_text('original')
            def inspect(source,episode,*args):
                if episode==2:raise ValueError('missing depth')
                return {},{},dict.fromkeys(['cam_0','cam_1','cam_2']),[0,2],None
            with patch.object(batch.converter,'inspect_recording',side_effect=inspect),patch.object(batch.converter,'main') as convert:
                result=batch.main(['--input',str(source),'--outputs-root',str(outputs),'--overwrite'])
            self.assertEqual(result,2);convert.assert_not_called();self.assertEqual((old/'keep').read_text(),'original')

    def test_six_episode_prompts_and_mask_failure_does_not_skip_later_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'dataset_20261021_161154';source.mkdir();outputs=root/'out'
            with patch.object(batch.converter,'inspect_recording',return_value=({}, {}, {'cam_0':{},'cam_1':{},'cam_2':{}},[0,2],None)),patch.object(batch.converter,'main') as convert,patch.object(batch.subprocess,'run',side_effect=[subprocess.CompletedProcess([],1)]+[subprocess.CompletedProcess([],0)]*5) as masks:
                result=batch.main(['--input',str(source),'--outputs-root',str(outputs),'--point-fraction','.125'])
            self.assertEqual(result,1);self.assertEqual(convert.call_count,6)
            for call in convert.call_args_list:
                self.assertEqual(call.args[0][call.args[0].index('--point-fraction')+1], '0.125')
            prompts=[call.args[0][call.args[0].index('--prompt')+1] for call in masks.call_args_list]
            self.assertEqual(prompts,['white drawer']*3+['black box','laptop','black monitor'])

    def test_named_inputs_discover_episodes_and_derive_prompts_including_typo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            laptop=self.named_source(root,'final_black_laptop',[0,7])
            suitcase=self.named_source(root,'final_orange_suitecase',[12])
            with patch.object(batch.converter,'inspect_recording',return_value=({}, {}, {'cam_0':{}},[0,2],None)),patch.object(batch.converter,'main') as convert,patch.object(batch.subprocess,'run',return_value=subprocess.CompletedProcess([],0)) as masks:
                result=batch.main(['--inputs',str(laptop),str(suitcase),'--outputs-root',str(root/'out')])
            self.assertEqual(result,0)
            prompts=[c.args[0][c.args[0].index('--prompt')+1] for c in masks.call_args_list]
            self.assertEqual(prompts,['black laptop','black laptop','orange suitcase'])
            outputs=[Path(c.args[0][c.args[0].index('--output')+1]).name for c in convert.call_args_list]
            self.assertEqual(outputs,['final_black_laptop_episode_000000','final_black_laptop_episode_000007','final_orange_suitecase_episode_000012'])
            for c in masks.call_args_list:
                self.assertNotIn('--prompt-points',c.args[0]);self.assertNotIn('--exclude-prompt',c.args[0])

    def test_resume_preserves_matching_conversion_and_resumes_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.named_source(root,'final_robot_dog',[7])
            output=root/'out/final_robot_dog_episode_000007';(output/'metadata').mkdir(parents=True)
            (output/'metadata/real_conversion.json').write_text(json.dumps(dict(source=str(source),episode=7,source_frames=[0,2])))
            (output/'metadata/mask_generation.json').write_text('{}')
            (output/'keep').write_text('original')
            with patch.object(batch.converter,'inspect_recording',return_value=({}, {}, {'cam_0':{}},[0,2],None)),patch.object(batch.converter,'main') as convert,patch.object(batch.subprocess,'run',return_value=subprocess.CompletedProcess([],0)) as masks:
                result=batch.main(['--inputs',str(source),'--outputs-root',str(root/'out'),'--resume'])
            self.assertEqual(result,0);convert.assert_not_called()
            self.assertIn('--resume',masks.call_args.args[0])
            self.assertEqual((output/'keep').read_text(),'original')

    def test_resume_incomplete_conversion_stops_without_removing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.named_source(root,'final_black_box',[0])
            output=root/'out/final_black_box_episode_000000';output.mkdir(parents=True)
            marker=output/'conversion_in_progress.json';marker.write_text('{}')
            with patch.object(batch.converter,'inspect_recording',return_value=({}, {}, {'cam_0':{}},[0,2],None)),patch.object(batch.converter,'main') as convert,patch.object(batch.subprocess,'run') as masks:
                result=batch.main(['--inputs',str(source),'--outputs-root',str(root/'out'),'--resume'])
            self.assertEqual(result,2);convert.assert_not_called();masks.assert_not_called()
            self.assertTrue(marker.exists())


if __name__=='__main__':unittest.main()
