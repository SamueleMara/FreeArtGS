"""Repair legacy real-scene frame names without re-encoding data or rebuilding geometry."""
import csv
import hashlib
import io
import json
import re
import shutil
import struct
from pathlib import Path

FRAME = re.compile(r"frame_(\d{6})(?!\d)")
JOURNAL = "metadata/frame_renumbering_backup"


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, data):
    temporary = path.with_name(path.name + '.renumber-tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def rename_string(value, mapping):
    def replace(match):
        old = int(match[1])
        if old not in mapping:
            raise ValueError(f'Unexpected frame reference: {match[0]}')
        return f'frame_{mapping[old]:06d}'
    return FRAME.sub(replace, value)


def rename_json(value, mapping):
    if isinstance(value, dict):
        result = {rename_string(k, mapping): rename_json(v, mapping) for k, v in value.items()}
        if 'endpoint_frame' in value:
            result['source_frame'] = value.get('source_frame', value['endpoint_frame'])
            result['endpoint_frame'] = mapping[value['endpoint_frame']]
        return result
    if isinstance(value, list):
        return [rename_json(v, mapping) for v in value]
    return rename_string(value, mapping) if isinstance(value, str) else value


def rename_images_binary(data, mapping):
    """Keep COLMAP IDs, poses, observations and tracks byte-for-byte unchanged."""
    stream = io.BytesIO(data)
    def read(size):
        result = stream.read(size)
        if len(result) != size:
            raise ValueError('Truncated COLMAP images.bin')
        return result
    header = read(8)
    result = bytearray(header)
    names = []
    for _ in range(struct.unpack('<Q', header)[0]):
        result.extend(read(64))
        name = bytearray()
        while (char := read(1)) != b'\0':
            name.extend(char)
        renamed = rename_string(name.decode(), mapping)
        names.append(renamed)
        result.extend(renamed.encode() + b'\0')
        count = read(8)
        result.extend(count)
        result.extend(read(struct.unpack('<Q', count)[0] * 24))
    if stream.read():
        raise ValueError('Unexpected trailing bytes in COLMAP images.bin')
    return bytes(result), names


def prepare(output):
    if (output / 'conversion_in_progress.json').exists():
        raise ValueError('RGB-D conversion is incomplete')
    metadata = json.loads((output / 'metadata/real_conversion.json').read_text())
    frames = metadata['source_frames']
    if not frames or frames != sorted(set(frames)):
        raise ValueError('Expected unique increasing source frame indices')
    mapping = dict(zip(frames, range(len(frames))))
    if metadata.get('frame_naming') == 'consecutive_zero_based':
        return None, {}
    camera_names = [c['name'] for c in json.loads((output / 'metadata/cameras.json').read_text())['cameras']]
    # Require the complete synchronized modalities before touching any filename.
    for folder, suffix in [('rgb', '.png'), ('depth_npy', '.npy'), ('depth_png', '.png'), ('mask', '.png')]:
        if folder == 'mask' and not (output / folder).exists():
            continue
        for camera in camera_names:
            actual = {p.name for p in (output / folder / camera).glob('*') if p.is_file()}
            expected = {f'frame_{frame:06d}{suffix}' for frame in frames}
            if actual != expected:
                raise ValueError(f'{folder}/{camera}: missing or unexpected frames')
    files = []
    updates = {}
    for path in sorted(output.rglob('*')):
        relative = path.relative_to(output)
        if relative.parts[:2] == ('metadata', 'raw') or relative.parts[0] == '.runtime' or path.is_dir():
            continue
        if path.is_symlink():
            raise ValueError(f'Refusing symlink: {relative}')
        if FRAME.search(path.name):
            renamed = path.with_name(rename_string(path.name, mapping))
            files.append(dict(old=str(relative), new=str(renamed.relative_to(output)), sha256=digest(path)))
        elif path.suffix == '.json':
            value = rename_json(json.loads(path.read_text()), mapping)
            if relative == Path('metadata/real_conversion.json'):
                value.update(output_frames=list(range(len(frames))), frame_naming='consecutive_zero_based',
                             timestamps='source_frame_index / recording FPS; not hardware timestamps')
            updates[str(relative)] = json_bytes(value)
        elif path.name == 'images.bin':
            updates[str(relative)], names = rename_images_binary(path.read_bytes(), mapping)
            for name in names:
                old_name = next((f'frame_{old:06d}' for old, new in mapping.items() if f'frame_{new:06d}' == Path(name).stem), None)
                if old_name is None or not (path.parents[2] / 'images' / Path(name).parent / (old_name + '.png')).is_file():
                    raise ValueError(f'Missing image referenced by {relative}: {name}')
        elif path.name == 'images.txt':
            updates[str(relative)] = rename_string(path.read_text(), mapping).encode()
        elif relative == Path('metadata/frame_timestamps.csv'):
            rows = list(csv.DictReader(io.StringIO(path.read_text())))
            if [int(row['frame_index']) for row in rows] != frames:
                raise ValueError('Timestamps do not match source frames')
            stream = io.StringIO(newline='')
            writer = csv.writer(stream)
            writer.writerow(['frame_index', 'source_frame_index', 'timestamp_s'])
            writer.writerows((mapping[int(row['frame_index'])], int(row['frame_index']), row['timestamp_s']) for row in rows)
            updates[str(relative)] = stream.getvalue().encode()
        elif path.name == 'README.txt':
            updates[str(relative)] = (rename_string(path.read_text(), mapping) +
                '\nFrame files are consecutive from zero. Original capture frame indices and timestamps are retained in metadata/real_conversion.json and metadata/frame_timestamps.csv.\n').encode()
    old_paths = {f['old'] for f in files}
    new_paths = {f['new'] for f in files}
    if len(new_paths) != len(files):
        raise ValueError('Frame renaming would collide')
    for name in new_paths - old_paths:
        if (output / name).exists():
            raise ValueError(f'Destination already exists: {name}')
    return dict(phase='stage', source_frames=frames, files=files, updates=list(updates)), updates


def renumber(output, dry_run=False):
    output = Path(output).expanduser().resolve()
    backup = output / JOURNAL
    journal_path = backup / 'journal.json'
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        print(f'Repair journal: {journal["phase"]}', flush=True)
    else:
        if backup.exists():
            raise ValueError(f'Incomplete repair preparation; inspect {backup} before retrying')
        journal, updates = prepare(output)
        if journal is None:
            print('Already uses consecutive zero-based frame names; nothing changed.', flush=True)
            return
        print(f'{len(journal["source_frames"])} frames/camera; {len(journal["files"])} data files; {len(updates)} reference files', flush=True)
        if dry_run:
            return
        (backup / 'files').mkdir(parents=True)
        for name, data in updates.items():
            for kind in ['originals', 'updates']:
                (backup / kind / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output / name, backup / 'originals' / name)
            (backup / 'updates' / name).write_bytes(data)
        atomic_write(journal_path, json_bytes(journal))
    if dry_run:
        return
    # Two phases avoid collisions such as old frame 2 becoming new frame 1,
    # while old frame 4 becomes new frame 2. The journal permits resuming.
    if journal['phase'] == 'stage':
        for index, item in enumerate(journal['files']):
            staged = backup / 'files' / str(index)
            if not staged.exists():
                (output / item['old']).rename(staged)
        journal['phase'] = 'install'
        atomic_write(journal_path, json_bytes(journal))
    if journal['phase'] == 'install':
        for index, item in enumerate(journal['files']):
            staged = backup / 'files' / str(index)
            destination = output / item['new']
            if staged.exists():
                if destination.exists():
                    raise FileExistsError(destination)
                staged.rename(destination)
        for name in journal['updates']:
            atomic_write(output / name, (backup / 'updates' / name).read_bytes())
        journal['phase'] = 'verify'
        atomic_write(journal_path, json_bytes(journal))
    for item in journal['files']:
        if digest(output / item['new']) != item['sha256']:
            raise ValueError(f'Data changed during renaming: {item["new"]}')
    for name in journal['updates']:
        if (output / name).read_bytes() != (backup / 'updates' / name).read_bytes():
            raise ValueError(f'Reference update mismatch: {name}')
    journal['phase'] = 'complete'
    atomic_write(journal_path, json_bytes(journal))
    print(f'Renumbered and SHA256-verified {len(journal["files"])} data files; reference backups: {backup}', flush=True)
