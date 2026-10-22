#!/usr/bin/env python3
"""Prepare the six real episodes with their explicitly assigned mask prompts."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

import convert_real_recording as converter
from point_sampling import point_fraction

PROMPTS = ('white drawer', 'white drawer', 'white drawer', 'black box', 'laptop', 'black monitor')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--outputs-root',type=Path,required=True)
    parser.add_argument('--episodes',type=int,nargs='+',choices=range(6),default=list(range(6)))
    parser.add_argument('--overwrite',action='store_true',help='Delete and rebuild selected named scene folders, only after all selected inputs validate')
    parser.add_argument('--dry-run',action='store_true',help='Validate inputs and print prompts/outputs; write nothing')
    parser.add_argument('--point-fraction',type=point_fraction,help='Fraction (0, 1] of valid depth points in endpoint PCDs/COLMAP; saved for masked export')
    args=parser.parse_args(argv)
    source=args.input.expanduser().resolve();root=args.outputs_root.expanduser().resolve()
    if source==root or source.is_relative_to(root) or root.is_relative_to(source):
        parser.error('Input and output must be separate, non-nested directories')
    match=re.search(r'\d{8}_\d{6}$',source.name)
    if not match:parser.error('Input folder must end with its YYYYMMDD_HHMMSS recording ID')
    plans=[];failures=[]
    for episode in dict.fromkeys(args.episodes):
        prompt=PROMPTS[episode]
        output=root/f'{prompt.replace(" ","_")}_{match.group()}_episode_{episode:06d}'
        try:
            _,_,cameras,frames,_=converter.inspect_recording(source,episode,[640,480],0)
            if output.exists() and not args.overwrite and not args.dry_run:
                raise FileExistsError(f'Output exists; use --overwrite: {output}')
            if output.is_symlink() or output.resolve().parent!=root:
                raise ValueError(f'Output must be a direct scene folder, not a symlink: {output}')
            print(f'episode {episode:06d}: {len(cameras)} cameras x {len(frames)} RGB-D frames; prompt={prompt!r}; output={output}',flush=True)
            plans.append((episode,prompt,output))
        except Exception as error:
            failures.append(f'episode {episode:06d}: {error}')
    if failures:
        print('\nInput validation failed. No output folders were deleted or changed.',file=sys.stderr)
        for failure in failures:print(failure,file=sys.stderr)
        return 2
    if args.dry_run:return 0
    failed=False
    for episode,prompt,output in plans:
        try:
            if output.exists():shutil.rmtree(output)
            convert_args=['--input',str(source),'--output',str(output),'--episode',str(episode)]
            if args.point_fraction is not None:
                convert_args += ['--point-fraction',str(args.point_fraction)]
            converter.main(convert_args)
            command=[sys.executable,'-B',str(Path(__file__).with_name('create_standard_masks.py')),
                     '--input',str(output),'--prompt',prompt,'--device','cuda']
            result=subprocess.run(command)
            if result.returncode:
                failed=True
                print(f'episode {episode:06d}: mask generation returned {result.returncode}; inspect its log and mask_generation.json',file=sys.stderr)
        except Exception as error:
            failed=True;print(f'episode {episode:06d}: {error}',file=sys.stderr)
    return 1 if failed else 0


if __name__=='__main__':sys.exit(main())
