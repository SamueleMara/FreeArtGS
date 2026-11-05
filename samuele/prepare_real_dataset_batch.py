#!/usr/bin/env python3
"""Prepare named real recordings, or the legacy six-episode capture, with SAM masks."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import convert_real_recording as converter
from point_sampling import point_fraction

PROMPTS = ('white drawer', 'white drawer', 'white drawer', 'black box', 'laptop', 'black monitor')


def recording_jobs(source, episodes, named):
    if named:
        label = source.name.removeprefix('final_')
        if not label or not re.fullmatch(r'[A-Za-z0-9_-]+', label):
            raise ValueError(f'Invalid object folder name: {source.name}')
        prompt = ' '.join(label.replace('_', ' ').split()).replace('suitecase', 'suitcase')
        rows = [json.loads(line) for line in (source/'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        available = {int(row['episode_index']) for row in rows}
        selected = sorted(available) if episodes is None else list(dict.fromkeys(episodes))
        if not selected or any(index < 0 or index not in available for index in selected):
            raise ValueError(f'Selected episodes do not exist in {source.name}: {selected}')
        return [(index,prompt,f'{source.name}_episode_{index:06d}') for index in selected]
    match = re.search(r'\d{8}_\d{6}$',source.name)
    if not match:
        raise ValueError('Legacy --input folder must end with YYYYMMDD_HHMMSS; use --inputs for named recordings')
    selected = list(range(6)) if episodes is None else list(dict.fromkeys(episodes))
    if any(index not in range(6) for index in selected):
        raise ValueError('Legacy recording supports episodes 0..5')
    return [(index,PROMPTS[index],f'{PROMPTS[index].replace(" ","_")}_{match.group()}_episode_{index:06d}') for index in selected]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input',type=Path,help='Legacy six-episode capture with explicitly assigned prompts')
    inputs.add_argument('--inputs',type=Path,nargs='+',help='Named native recordings: discover all episodes and derive prompts from folder names')
    parser.add_argument('--outputs-root',type=Path,required=True)
    parser.add_argument('--episodes',type=int,nargs='+',help='Optional episode selection; default all episodes')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--overwrite',action='store_true',help='Delete and rebuild selected named scene folders, only after all selected inputs validate')
    modes.add_argument('--resume',action='store_true',help='Reuse completed matching conversions and resume matching mask manifests')
    parser.add_argument('--dry-run',action='store_true',help='Validate inputs and print prompts/outputs; write nothing')
    parser.add_argument('--point-fraction',type=point_fraction,help='Fraction (0, 1] of valid depth points in endpoint PCDs/COLMAP; saved for masked export')
    args=parser.parse_args(argv)
    root=args.outputs_root.expanduser().resolve()
    sources=list(dict.fromkeys(p.expanduser().resolve() for p in (args.inputs or [args.input])))
    jobs=[];failures=[]
    for source in sources:
        try:
            if source==root or source.is_relative_to(root) or root.is_relative_to(source):
                raise ValueError('Input and output must be separate, non-nested directories')
            jobs.extend((source,*job) for job in recording_jobs(source,args.episodes,args.inputs is not None))
        except Exception as error:
            failures.append(f'{source.name}: {error}')
    plans=[];outputs=set()
    for source,episode,prompt,name in jobs:
        output=root/name
        try:
            _,_,cameras,frames,_=converter.inspect_recording(source,episode,[640,480],0)
            if output in outputs:
                raise ValueError(f'Duplicate output scene name: {output}')
            outputs.add(output)
            if output.is_symlink() or output.resolve().parent!=root:
                raise ValueError(f'Output must be a direct scene folder, not a symlink: {output}')
            if output.exists() and args.resume:
                if (output/'conversion_in_progress.json').exists():
                    raise ValueError(f'Conversion incomplete; inspect or rebuild explicitly with --overwrite: {output}')
                metadata=json.loads((output/'metadata/real_conversion.json').read_text())
                if (Path(metadata['source']).resolve()!=source or metadata['episode']!=episode or metadata['source_frames']!=frames):
                    raise ValueError(f'Existing conversion source/episode/frame selection differs: {output}')
            elif output.exists() and not args.overwrite and not args.dry_run:
                raise FileExistsError(f'Output exists; use --resume or --overwrite: {output}')
            print(f'{source.name} episode {episode:06d}: {len(cameras)} cameras x {len(frames)} RGB-D frames; prompt={prompt!r}; output={output}',flush=True)
            plans.append((source,episode,prompt,output))
        except Exception as error:
            failures.append(f'{source.name} episode {episode:06d}: {error}')
    if failures:
        print('\nInput validation failed. No output folders were deleted or changed.',file=sys.stderr)
        for failure in failures:print(failure,file=sys.stderr)
        return 2
    print(f'Validated {len(plans)} episodes across {len(sources)} recordings.',flush=True)
    if args.dry_run:return 0
    failed=False
    for source,episode,prompt,output in plans:
        try:
            if not (args.resume and output.exists()):
                if output.exists():shutil.rmtree(output)
                convert_args=['--input',str(source),'--output',str(output),'--episode',str(episode)]
                if args.point_fraction is not None:
                    convert_args += ['--point-fraction',str(args.point_fraction)]
                converter.main(convert_args)
            command=[sys.executable,'-B',str(Path(__file__).with_name('create_standard_masks.py')),
                     '--input',str(output),'--prompt',prompt,'--device','cuda']
            if args.resume and (output/'metadata/mask_generation.json').is_file():
                command += ['--resume']
            if args.point_fraction is not None:
                command += ['--point-fraction',str(args.point_fraction)]
            result=subprocess.run(command)
            if result.returncode:
                failed=True
                print(f'{source.name} episode {episode:06d}: mask generation returned {result.returncode}; inspect its log and mask_generation.json',file=sys.stderr)
        except Exception as error:
            failed=True;print(f'{source.name} episode {episode:06d}: {error}',file=sys.stderr)
    return 1 if failed else 0


if __name__=='__main__':sys.exit(main())
