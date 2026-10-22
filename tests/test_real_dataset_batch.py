from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'samuele'))
import prepare_real_dataset_batch as batch


class RealBatchTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
