#!/usr/bin/env python3
"""GroundingDINO + SAM2 foreground masks for rgb/cam_NNN/frame_NNNNNN.png."""
import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from utils.local_storage import configure_runtime

DEFAULT_MODEL_ROOT = Path.home() / "Grounded-SAM-2-autodistill"


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def list_frames(dataset, limit=0):
    frames = []
    for camera in sorted((dataset / "rgb").glob("cam_*")):
        if not camera.is_dir() or not re.fullmatch(r"cam_\d+", camera.name):
            continue
        files = sorted(p for p in camera.glob("frame_*.png") if re.fullmatch(r"frame_\d+", p.stem))
        frames.extend(files[:limit] if limit else files)
    if not frames:
        raise ValueError(f"No RGB frames found under {dataset / 'rgb/cam_NNN'}")
    return frames


def check_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != shape or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all():
        raise ValueError("Masks must be uint8 0/255 with the original RGB dimensions")
    return mask


def load_segmenter(args):
    root = args.model_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "grounding_dino"))
    # All weights already exist locally; use the read-only BERT snapshot directly.
    snapshots = Path.home() / ".cache/huggingface/hub/models--bert-base-uncased/snapshots"
    bert = next((p for p in sorted(snapshots.glob("*")) if (p / "config.json").is_file() and
                 ((p / "model.safetensors").is_file() or (p / "pytorch_model.bin").is_file())), None)
    if bert is None:
        raise FileNotFoundError(f"No cached bert-base-uncased model: {snapshots}")
    config = root / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_weights = root / "gdino_checkpoints/groundingdino_swint_ogc.pth"
    sam_weights = root / "checkpoints/sam2.1_hiera_base_plus.pt"
    for path in [config, dino_weights, sam_weights]:
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use the grounded-sam2 environment; CPU requires explicit --device cpu")
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.models import build_model
    from groundingdino.util.inference import load_image, predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    configuration = SLConfig.fromfile(str(config))
    configuration.device = args.device
    configuration.text_encoder_type = str(bert)
    print(f"Loading local models on {args.device}; no downloads", flush=True)
    dino = build_model(configuration)
    dino.load_state_dict(clean_state_dict(torch.load(dino_weights, map_location="cpu")["model"]), strict=False)
    dino = dino.eval().to(args.device)
    predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", str(sam_weights), device=args.device))

    def segment(path, prompt):
        rgb, tensor = load_image(str(path))
        height, width = rgb.shape[:2]
        combined = np.zeros((height, width), dtype=np.uint8)
        with torch.inference_mode():
            boxes, confidence, _ = predict(dino, tensor, prompt, args.box_thr, args.text_thr, device=args.device)
            boxes = boxes.detach().cpu().numpy() * [width, height, width, height]
            xyxy = np.column_stack([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2])
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else contextlib.nullcontext()
            with autocast:
                if len(xyxy):
                    predictor.set_image(rgb)  # SAM2 expects RGB, not OpenCV BGR.
                for box in xyxy:
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                    if len(scores):
                        combined |= (masks[int(np.argmax(scores))] > .5).astype(np.uint8) * 255
        return combined, dict(detections=len(boxes), max_detection_confidence=float(confidence.max().item()) if len(confidence) else None)

    return segment


def generate(args, segment=None):
    dataset = args.input.expanduser().resolve()
    frames = list_frames(dataset, args.max_frames)
    settings = dict(prompt=args.prompt.strip(), box_threshold=args.box_thr, text_threshold=args.text_thr,
                    sam_model="sam2.1_hiera_base_plus", model_root=str(args.model_root.expanduser().resolve()))
    manifest_file = dataset / "metadata/mask_generation.json"
    previous = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    if previous and previous.get("settings") != settings and not args.overwrite:
        raise ValueError("Mask settings/prompt changed. Use --overwrite to regenerate masks explicitly")
    print(f"{len(frames)} RGB frames; prompt={settings['prompt']!r}; masks -> {dataset / 'mask'}", flush=True)
    if args.dry_run:
        return dict(requested=len(frames))
    for path in [dataset / "mask", manifest_file]:
        if not path.resolve().is_relative_to(dataset):
            raise ValueError(f"Output escapes the dataset: {path}")
    configure_runtime(dataset)
    for directory in [dataset / "mask", dataset / "metadata"]:
        directory.mkdir(parents=True, exist_ok=True)
    records = {} if args.overwrite else previous.get("frames", {})
    # Refuse unknown masks before loading models or changing any files.
    for path in frames:
        relative = path.relative_to(dataset / "rgb")
        target = dataset / "mask" / relative
        if target.exists() and str(relative) not in records and not args.overwrite:
            raise ValueError(f"Existing mask has no generation record; use --overwrite: {target}")
    manifest = dict(settings=settings, frames=records, requested=len(frames), completed=0, empty=0, skipped=0)
    for index, path in enumerate(frames):
        relative = path.relative_to(dataset / "rgb")
        target = dataset / "mask" / relative
        if not target.resolve().is_relative_to(dataset):
            raise ValueError(f"Mask output escapes the dataset: {target}")
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            shape = (image.height, image.width)
        old = records.get(str(relative), {})
        if target.exists() and not args.overwrite:
            if old.get("rgb_sha256") != source_hash:
                raise ValueError(f"RGB changed since masking; use --overwrite: {path}")
            with Image.open(target) as image:
                mask = check_mask(np.asarray(image), shape)
            details = old
            manifest["skipped"] += 1
        else:
            if segment is None:
                segment = load_segmenter(args)
            mask, details = segment(path, settings["prompt"])
            mask = check_mask(mask, shape)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".png.tmp")
            Image.fromarray(mask).save(temporary, format="PNG")
            temporary.replace(target)
        count = int(np.count_nonzero(mask))
        records[str(relative)] = dict(details, rgb_sha256=source_hash, foreground_pixels=count,
                                      foreground_fraction=count / mask.size, status="nonempty" if count else "empty")
        manifest["completed"] += 1
        manifest["empty"] += count == 0
        write_json(manifest_file, manifest)
        if index % 25 == 0 or index == len(frames) - 1:
            print(f"{index + 1}/{len(frames)} masks; empty={manifest['empty']}; reused={manifest['skipped']}", flush=True)
    if not args.max_frames and not manifest["empty"] and all((dataset / model).is_dir() for model in ["colmap", "colmap_final"]):
        from colmap_masking import apply_scene_masks
        apply_scene_masks(dataset)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Converted scene containing rgb/cam_NNN/*.png")
    parser.add_argument("--prompt", required=True, help='Object description, e.g. "red fridge" or "refrigerator"')
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--box-thr", type=float, default=.25)
    parser.add_argument("--text-thr", type=float, default=.25)
    parser.add_argument("--max-frames", type=int, default=0, help="Per-camera limit; zero means all frames")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace masks or change the prompt/settings")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.prompt.strip() or args.max_frames < 0 or not 0 <= args.box_thr <= 1 or not 0 <= args.text_thr <= 1:
        parser.error("Prompt must be nonempty, frame limit nonnegative, and thresholds in [0, 1]")
    result = generate(args)
    if result.get("empty", 0):
        print(f"WARNING: {result['empty']} empty masks; inspect metadata/mask_generation.json and adjust the prompt if needed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
