#!/usr/bin/env python3
"""Export measured robot poses from each real episode's Parquet to NumPy archives."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from convert_real_recording import transform
from robot_mask_guidance import read_parquet_rows

POSE_FIELDS=('pos_x','pos_y','pos_z','rotvec_x','rotvec_y','rotvec_z')
EXPORT_FILES=('poses_full.npz','poses_rgbd.npz','manifest.json')


def calibration_for_scene(scene,mapping,recorded_frame):
    entry=dict(mapping.get('defaults',{}))
    datasets=mapping.get('datasets',{})
    name=Path(scene).name
    entry.update(datasets.get(name.rsplit('_episode_',1)[0],{}))
    entry.update(datasets.get(name,{}))
    frame=entry.get('recorded_frame',recorded_frame)
    matrix=entry.get('T_ee_tcp')
    return frame,None if matrix is None else transform(matrix,'T_ee_tcp')


def build_arrays(rows,info,conversion,recorded_frame='unknown',ee_to_tcp=None):
    names=info.get('state_names',[])
    if any(field not in names for field in POSE_FIELDS):
        raise ValueError('Raw info.json must identify Cartesian positions and rotation-vector state fields')
    indices=[names.index(field) for field in POSE_FIELDS]
    if not rows:raise ValueError('Parquet contains no observations')
    frames=np.asarray([row['frame_index'] for row in rows])
    if frames.ndim!=1 or not np.issubdtype(frames.dtype,np.integer) or len(np.unique(frames))!=len(frames) or np.any(frames<0):
        raise ValueError('Parquet frame indices must be unique nonnegative integers')
    order=np.argsort(frames)
    rows=[rows[int(i)] for i in order];frames=frames[order].astype(np.int64)
    timestamps=np.array([row['timestamp'] for row in rows],dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps)<0):
        raise ValueError('Recorded timestamps must be finite and nondecreasing')
    episode=int(conversion['episode'])
    if any(row['episode_index']!=episode for row in rows):
        raise ValueError('Parquet contains observations from a different episode')
    try:poses=np.array([[row['observation.state'][i] for i in indices] for row in rows],dtype=np.float64)
    except (IndexError,TypeError) as error:raise ValueError('State vectors do not match raw pose field definitions') from error
    if not np.isfinite(poses).all():raise ValueError('Measured robot poses contain nonfinite values')
    recorded=np.repeat(np.eye(4)[None],len(poses),axis=0)
    recorded[:,:3,:3]=Rotation.from_rotvec(poses[:,3:]).as_matrix()
    recorded[:,:3,3]=poses[:,:3]
    full=dict(frame_index=frames,source_frame_index=frames.copy(),timestamp_s=timestamps,
        episode_index=np.full(len(frames),episode,dtype=np.int64),recorded_position_m=poses[:,:3],
        recorded_rotvec_rad=poses[:,3:],recorded_transform=recorded)
    if recorded_frame not in {'unknown','ee','tcp'}:raise ValueError('recorded_frame must be unknown, ee, or tcp')
    if ee_to_tcp is not None:
        ee_to_tcp=transform(ee_to_tcp,'T_ee_tcp')
        if recorded_frame=='unknown':raise ValueError('Declare --recorded-frame ee or tcp before deriving the other pose')
    if recorded_frame=='ee':
        full['ee_transform']=recorded
        if ee_to_tcp is not None:full['tcp_transform']=recorded@ee_to_tcp
    elif recorded_frame=='tcp':
        full['tcp_transform']=recorded
        if ee_to_tcp is not None:full['ee_transform']=recorded@np.linalg.inv(ee_to_tcp)
    for frame in ['ee','tcp']:
        if frame+'_transform' in full:
            full[frame+'_position_m']=full[frame+'_transform'][:,:3,3]
    source_frames=conversion['source_frames']
    output_frames=conversion.get('output_frames',source_frames)
    if not source_frames or len(source_frames)!=len(output_frames):
        raise ValueError('Invalid RGB-D output/source frame mapping')
    for values in [source_frames,output_frames]:
        data=np.asarray(values)
        if data.ndim!=1 or not np.issubdtype(data.dtype,np.integer) or np.any(data<0) or len(np.unique(data))!=len(data):
            raise ValueError('RGB-D output/source frame indices must be unique nonnegative integers')
    lookup={int(frame):i for i,frame in enumerate(frames)}
    missing=[frame for frame in source_frames if frame not in lookup]
    if missing:raise ValueError(f'RGB-D source frames absent from Parquet: {missing[:10]}')
    selected=np.array([lookup[frame] for frame in source_frames],dtype=np.int64)
    aligned={name:value[selected] for name,value in full.items()}
    aligned['frame_index']=np.asarray(output_frames,dtype=np.int64)
    return full,aligned


def export_scene(scene,parquet_python,recorded_frame,ee_to_tcp,overwrite=False,dry_run=False,output_dir=None):
    scene=Path(scene).expanduser().resolve()
    conversion_file=scene/'metadata/real_conversion.json'
    conversion=json.loads(conversion_file.read_text())
    if (scene/'conversion_in_progress.json').exists():raise ValueError(f'Incomplete conversion: {scene}')
    raw=scene/'metadata/raw';parquet=raw/f'episode_{conversion["episode"]:06d}.parquet'
    info=json.loads((raw/'info.json').read_text())
    rows=read_parquet_rows(parquet,parquet_python,['timestamp','frame_index','episode_index','observation.state'])
    full,aligned=build_arrays(rows,info,conversion,recorded_frame,ee_to_tcp)
    output=Path(output_dir).resolve() if output_dir else scene/'metadata/robot_poses'
    if output_dir is None and not output.resolve().is_relative_to(scene):raise ValueError(f'Robot pose output escapes scene: {output}')
    if not overwrite and any((output/name).exists() for name in EXPORT_FILES):
        raise FileExistsError(f'Robot pose exports exist; use --overwrite: {output}')
    manifest=dict(source_parquet=str(parquet),source_parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
        conversion_sha256=hashlib.sha256(conversion_file.read_bytes()).hexdigest(),
        episode=conversion['episode'],recorded_frame=recorded_frame,pose_fields=list(POSE_FIELDS),
        coordinate_frame=conversion.get('coordinate_frame','unspecified_recording_frame'),
        position_unit='meters',orientation='rotation_vector_radians',transform_convention='T_base_frame maps frame-local points into recording base coordinates',
        timestamp_source='recorded timestamp column, seconds relative to episode; no hardware synchronization claim',
        full_rows=len(full['frame_index']),rgbd_rows=len(aligned['frame_index']),
        arrays=list(full),ee_available='ee_position_m' in full,tcp_available='tcp_position_m' in full,
        ee_to_tcp=None if ee_to_tcp is None else np.asarray(ee_to_tcp).tolist(),
        tcp_calibration_note='Only an explicitly supplied rigid transform is used; masking offsets are not TCP calibration',
        absent_arrays='Unavailable EE/TCP arrays are omitted, never filled with zeros or copied under a second label')
    if not dry_run:
        output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.pose-export-',dir=output) as temp:
            stage=Path(temp)
            np.savez_compressed(stage/'poses_full.npz',**full)
            np.savez_compressed(stage/'poses_rgbd.npz',**aligned)
            (stage/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
            for name in EXPORT_FILES:(stage/name).replace(output/name)
    print(f'{scene.name}: {manifest["full_rows"]} recorded / {manifest["rgbd_rows"]} RGB-D rows; EE={manifest["ee_available"]}, TCP={manifest["tcp_available"]}; {output}',flush=True)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--root',type=Path,help='Standard real output root; all direct episode folders are processed')
    inputs.add_argument('--scene',type=Path,help='One converted real scene')
    parser.add_argument('--parquet-python',type=Path,default=Path(sys.executable),help='Installed Python with polars, used for CPU Parquet decoding only')
    parser.add_argument('--recorded-frame',choices=['unknown','ee','tcp'],default='unknown',help='Meaning of observation.state pose; default does not assume EE or TCP')
    tool=parser.add_mutually_exclusive_group()
    tool.add_argument('--ee-to-tcp',type=Path,help='JSON containing T_ee_tcp as a 4x4 matrix, translations in meters')
    tool.add_argument('--tcp-offset-m',type=float,nargs=3,metavar=('X','Y','Z'),help='Explicit EE-local TCP translation; TCP orientation equals EE orientation')
    tool.add_argument('--calibration-map',type=Path,help='JSON with defaults and per-recording/per-scene recorded_frame and T_ee_tcp entries')
    parser.add_argument('--output-dir',type=Path,help='Alternative export directory, only valid with --scene')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if args.output_dir and not args.scene:parser.error('--output-dir requires --scene')
    matrix=None
    if args.ee_to_tcp:
        data=json.loads(args.ee_to_tcp.read_text())
        matrix=transform(data['T_ee_tcp'] if isinstance(data,dict) else data,'T_ee_tcp')
    elif args.tcp_offset_m is not None:
        matrix=np.eye(4);matrix[:3,3]=args.tcp_offset_m;matrix=transform(matrix,'T_ee_tcp')
    if matrix is not None and args.recorded_frame=='unknown':parser.error('Specify --recorded-frame ee or tcp when supplying EE→TCP calibration')
    mapping=json.loads(args.calibration_map.read_text()) if args.calibration_map else None
    root=args.root.expanduser().resolve() if args.root else None
    scenes=[args.scene] if args.scene else sorted(p.parent.parent for p in root.glob('*/metadata/real_conversion.json'))
    if not scenes:parser.error('No converted real scenes found')
    failures=[]
    for scene in scenes:
        try:
            if root and (scene.is_symlink() or scene.resolve().parent!=root):
                raise ValueError(f'Export scene escapes the selected real output root: {scene}')
            frame,calibration=calibration_for_scene(scene,mapping,args.recorded_frame) if mapping is not None else (args.recorded_frame,matrix)
            export_scene(scene,args.parquet_python,frame,calibration,args.overwrite,args.dry_run,args.output_dir)
        except Exception as error:
            failures.append(str(scene));print(f'FAILED {Path(scene).name}: {error}',file=sys.stderr,flush=True)
    print(f'{len(scenes)-len(failures)} scenes exported/validated; {len(failures)} failures.',flush=True)
    return int(bool(failures))


if __name__=='__main__':sys.exit(main())
