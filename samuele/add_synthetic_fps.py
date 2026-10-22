#!/usr/bin/env python3
"""Backfill synthetic dataset FPS from video headers, with an explicit estimate fallback."""
import argparse
import json
import math
import os
from pathlib import Path
import re

import cv2
import yaml


def discover(root):
    scenes = []
    skip = {'rgb', 'depth_npy', 'depth_png', 'mask', 'colmap', 'colmap_final',
            'pointcloud', 'metadata', 'video_rgb', 'real', 'hoi'}
    for directory, children, _ in os.walk(root):
        scene = Path(directory)
        if (scene/'metadata/job_config_resolved.yaml').is_file():
            scenes.append(scene)
            children[:] = []
        else:
            children[:] = [name for name in children if name not in skip and not name.startswith('.')]
    return sorted(scenes)


def prepare(scene, expected_fps):
    scene = scene.resolve()
    metadata = scene/'metadata'
    config_path = metadata/'job_config_resolved.yaml'
    config_text = config_path.read_text()
    config = yaml.safe_load(config_text)
    cameras_path = metadata/'cameras.json'
    cameras = json.loads(cameras_path.read_text())
    rates = {}
    for video in sorted((scene/'video_rgb').glob('*.mp4')):
        cap = cv2.VideoCapture(str(video))
        try:
            rate = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not math.isfinite(rate) or rate <= 0:
                raise ValueError(f'Cannot read video FPS: {video}')
        finally:
            cap.release()
        if not math.isclose(rate, expected_fps, abs_tol=.01, rel_tol=0):
            raise ValueError(f'Video reports {rate} FPS, expected {expected_fps}: {video}')
        rates[video.name] = rate
    fps = sum(rates.values())/len(rates) if rates else expected_fps
    timing_path = metadata/'timing.json'
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing.update(fps=fps, fps_source='video_metadata' if rates else 'provided_video_estimate',
                  frame_interval_s=1/fps, num_frames=config.get('num_frames_resolved') or config.get('num_frames'),
                  video_fps_by_file=rates, reference_video_fps=expected_fps)
    cameras.update(fps=fps, fps_source=timing['fps_source'])
    # fps is a schedule input, mutually exclusive with num_frames. Keep it intact.
    if re.search(r'^fps_resolved:', config_text, flags=re.M):
        config_text = re.sub(r'^fps_resolved:.*$', f'fps_resolved: {fps}', config_text, flags=re.M)
    else:
        config_text = config_text.rstrip()+f'\nfps_resolved: {fps}\n'
    updates = {cameras_path:json.dumps(cameras, indent=2)+'\n',
               timing_path:json.dumps(timing, indent=2)+'\n', config_path:config_text}
    for path in updates:
        if not path.resolve().is_relative_to(scene):
            raise ValueError(f'Metadata path escapes scene: {path}')
    return updates, fps, len(rates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--fps', type=float, default=59., help='Expected video FPS and estimate for scenes without videos (default 59)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('FPS must be finite and positive')
    scenes = discover(args.root.expanduser().resolve())
    if not scenes:
        parser.error('No synthetic scenes with metadata/job_config_resolved.yaml found')
    plans = []
    for scene in scenes:
        updates, fps, videos = prepare(scene, args.fps)
        plans.append(updates)
        print(f'{scene}: fps={fps:.2f}; {videos} videos checked'+(' (provided estimate)' if not videos else ''), flush=True)
    if not args.dry_run:
        for updates in plans:
            for path, text in updates.items():
                if path.exists() and path.read_text() == text:
                    continue
                temporary = path.with_suffix(path.suffix+'.fps-tmp')
                temporary.write_text(text)
                temporary.replace(path)
    print(f'{"Validated" if args.dry_run else "Updated"} {len(scenes)} synthetic scenes.')


if __name__ == '__main__': main()
