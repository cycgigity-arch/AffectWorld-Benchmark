"""Experimental five-model Task 1 video score for a small EWM-Bench trial.

The score follows the three components and arithmetic mean in the NAACL paper.
The embedded trial protocol (or an optional JSON override) fixes choices
omitted by the paper. Results are not a paper-final benchmark score until the
protocol and automatic masks are audited.
The script scores each video on its own; original state1 videos are useful
calibration examples, while generated futures are the actual benchmark targets.
The VFI worker is included in this file and runs under --vfi-python.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SAM2 = "facebook/sam2.1-hiera-small"
DEFAULT_DINO = "facebook/dinov2-small"
DEFAULT_DEPTH = "depth-anything/Depth-Anything-V2-Small-hf"
TRIAL_PROTOCOL = {
    "gamma": 0.02,
    "lambda_depth": 5.0,
    "sample_fps": 2.0,
    "max_frames": 9,
    "resize_width": 512,
    "max_objects": 2,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--references-json", type=Path)
    inputs.add_argument("--video", type=Path, action="append", help="Score a standalone, already-cut video; may be repeated")
    parser.add_argument("--protocol", type=Path, help="JSON override; defaults to embedded experimental trial values")
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--allow-remote-source", action="store_true")
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--kind", choices=("reference", "generated", "both"))
    parser.add_argument("--output-dir", type=Path, default=Path("task1_score_results"))
    parser.add_argument("--limit", type=int, help="Maximum samples; defaults to 3 for references or all --video inputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-model", default=DEFAULT_SAM2)
    parser.add_argument("--dino-model", default=DEFAULT_DINO)
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH)
    parser.add_argument("--vfi-python", type=Path)
    parser.add_argument("--vfi-repo", type=Path)
    parser.add_argument("--vfi-model", choices=("VFIMamba_S", "VFIMamba"), default="VFIMamba_S")
    parser.add_argument("--check-only", action="store_true", help="Decode sampled frames and validate inputs without loading models")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.video:
        if args.kind or args.video_root or args.allow_remote_source or args.generated_dir:
            parser.error("--video cannot be combined with --kind or reference/generated directory options")
    else:
        args.kind = args.kind or "reference"
        if args.kind in {"reference", "both"} and not (args.video_root or args.allow_remote_source):
            parser.error("reference scoring needs --video-root or --allow-remote-source")
        if args.kind in {"generated", "both"} and not args.generated_dir:
            parser.error("generated scoring needs --generated-dir")
    if not args.check_only and (args.vfi_python is None or args.vfi_repo is None):
        parser.error("full scoring needs --vfi-python and --vfi-repo")
    return args


def load_protocol(path: Path | None) -> dict[str, float | int]:
    raw = json.loads(path.read_text(encoding="utf-8")) if path is not None else dict(TRIAL_PROTOCOL)
    keys = {"gamma", "lambda_depth", "sample_fps", "max_frames", "resize_width", "max_objects"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"protocol must contain exactly: {', '.join(sorted(keys))}")
    result = {key: float(raw[key]) for key in keys}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("protocol values must be finite")
    if result["lambda_depth"] < 0:
        raise ValueError("lambda_depth must be nonnegative")
    if result["gamma"] <= 0 or result["sample_fps"] <= 0:
        raise ValueError("gamma and sample_fps must be positive")
    for key, minimum in (("max_frames", 3), ("resize_width", 128), ("max_objects", 1)):
        value = result[key]
        if value != int(value) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        result[key] = int(value)
    return result


def load_references(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("references JSON must be a nonempty list")
    seen = set()
    result = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            raise ValueError("reference rows must be objects")
        key = row.get("video_key")
        state0 = row.get("original_state0") or {}
        state1 = row.get("original_state1") or {}
        if not isinstance(key, str) or key in seen or not key.startswith("videos/"):
            raise ValueError(f"invalid or duplicate video_key: {key}")
        try:
            before = state0["interval"]
            after = state1["interval"]
            end0 = float(before["end_percent"])
            start1 = float(after["start_percent"])
            end1 = float(after["end_percent"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid state interval for {key}") from error
        if not (0 <= end0 <= start1 < end1 <= 100):
            raise ValueError(f"non-chronological state1 interval for {key}")
        if not isinstance(row.get("expected_output_basename"), str):
            raise ValueError(f"missing generated filename for {key}")
        seen.add(key)
        result.append(row)
    return result


def resolve_original(root: Path, video_key: str) -> Path:
    parts = Path(video_key).parts
    direct = root.joinpath(*parts[1:])
    if direct.is_file():
        return direct
    original_name = parts[-1]
    stem = Path(original_name).stem
    marker = stem.find("_clip_")
    alternatives = [original_name]
    if marker >= 0:
        prefix = stem[:marker]
        for index in range(1, len(prefix)):
            left, right = prefix[:index], prefix[index:]
            if right == "_" + left:
                alternatives.append(left + stem[marker:] + Path(original_name).suffix)
                break
    group_dir = root / parts[1]
    matches = [path for name in alternatives for path in group_dir.rglob(name)] if group_dir.is_dir() else []
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise FileNotFoundError(f"expected one source for {video_key}, found {len(unique)}")
    return unique[0]


def run_checked(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command[:4])} failed: {result.stderr[-1200:]}")
    return result


def duration_seconds(source: str) -> float:
    result = run_checked(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", source], timeout=120)
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration: {source}")
    return duration


def sample_specs(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        key = row["video_key"]
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        if args.kind in {"reference", "both"}:
            if args.video_root:
                source = str(resolve_original(args.video_root, key))
            else:
                source = row.get("video_url")
                if not isinstance(source, str) or not source.startswith("https://"):
                    raise ValueError(f"missing HTTPS source URL for {key}")
            samples.append({"id": digest + "_reference", "video_key": key, "kind": "original_state1", "source": source,
                            "interval_percent": row["original_state1"]["interval"]})
        if args.kind in {"generated", "both"}:
            name = Path(row["expected_output_basename"]).name
            source = args.generated_dir / name
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"generated video missing: {source}")
            samples.append({"id": digest + "_generated", "video_key": key, "kind": "generated_future", "source": str(source)})
    return samples


def standalone_specs(paths: list[Path], limit: int) -> list[dict[str, Any]]:
    samples = []
    seen = set()
    for path in paths[:limit]:
        source = path.resolve()
        if source in seen:
            raise ValueError(f"duplicate standalone video: {source}")
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"standalone video missing or empty: {source}")
        digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
        samples.append({"id": digest + "_standalone", "video_key": f"standalone/{digest}/{source.name}",
                        "kind": "standalone_video", "source": str(source)})
        seen.add(source)
    return samples


def extract_frames(sample: dict[str, Any], destination: Path, protocol: dict[str, float | int]) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=False)
    source = sample["source"]
    duration = duration_seconds(source)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if sample["kind"] == "original_state1":
        interval = sample["interval_percent"]
        start = duration * float(interval["start_percent"]) / 100
        length = duration * (float(interval["end_percent"]) - float(interval["start_percent"])) / 100
        command += ["-ss", f"{start:.6f}", "-i", source, "-t", f"{length:.6f}"]
        sample["cut_seconds"] = {"start": start, "duration": length}
    else:
        command += ["-i", source]
    command += ["-vf", f"fps={protocol['sample_fps']},scale={protocol['resize_width']}:-2",
                "-frames:v", str(protocol["max_frames"]), "-q:v", "2", "-start_number", "0", str(destination / "%06d.jpg")]
    run_checked(command, timeout=1800)
    frames = sorted(destination.glob("*.jpg"))
    if len(frames) < 3:
        raise ValueError(f"fewer than 3 decoded frames: {source}")
    return frames


def choose_masks(candidates: list[dict[str, Any]], height: int, width: int, limit: int) -> list[dict[str, Any]]:
    scored = []
    for candidate in candidates:
        area = int(candidate.get("area", 0))
        fraction = area / (height * width)
        if not 0.01 <= fraction <= 0.60:
            continue
        x, y, bw, bh = candidate["bbox"]
        center_distance = math.hypot((x + bw / 2) / width - 0.5, (y + bh / 2) / height - 0.5)
        quality = float(candidate.get("predicted_iou", 0.7)) * float(candidate.get("stability_score", 0.7))
        score = quality * math.sqrt(fraction) * max(0.1, 1 - center_distance)
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    for score, candidate in scored:
        mask = candidate["segmentation"]
        if any((mask & old["segmentation"]).sum() / max(1, mask.sum()) > 0.5 for old in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) == limit:
            break
    if not chosen:
        raise ValueError("SAM2 found no suitable subject/object mask on the first frame")
    return chosen


def clear_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def avoid_sam2_repo_shadowing() -> None:
    """Keep a sibling SAM2 checkout from masking its installed Python package."""
    script_dir = Path(__file__).resolve().parent
    if not (script_dir / "sam2" / "sam2" / "build_sam.py").is_file():
        return
    sys.path[:] = [entry for entry in sys.path
                   if Path(entry or os.getcwd()).resolve() != script_dir]


def track_sam2(frames: list[Path], work: Path, model_id: str, device: str, max_objects: int) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf, build_sam2_video_predictor_hf

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"cannot read {frames[0]}")
    image = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    image_model = build_sam2_hf(model_id, device=device)
    generator = SAM2AutomaticMaskGenerator(image_model, points_per_side=16, points_per_batch=32)
    with torch.inference_mode():
        candidates = generator.generate(image)
    selected = choose_masks(candidates, image.shape[0], image.shape[1], max_objects)
    initial = [np.asarray(item["segmentation"], dtype=bool).copy() for item in selected]
    selection = [{"area_fraction": round(float(mask.mean()), 4), "bbox_xywh": list(map(int, item["bbox"]))}
                 for mask, item in zip(initial, selected)]
    preview = first.copy()
    colors = ((0, 210, 255), (230, 90, 30), (65, 220, 85))
    for obj_id, mask in enumerate(initial):
        color = np.asarray(colors[obj_id % len(colors)], dtype=np.float32)
        preview[mask] = (0.55 * preview[mask] + 0.45 * color).astype(np.uint8)
    preview_path = work / "sam2_first_frame_masks.jpg"
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"could not save SAM2 preview: {preview_path}")
    del generator, image_model, candidates, selected
    clear_gpu()

    mask_dir = work / "masks"
    mask_dir.mkdir()
    predictor = build_sam2_video_predictor_hf(model_id, device=device)
    state = predictor.init_state(video_path=str(frames[0].parent), offload_video_to_cpu=True, offload_state_to_cpu=True)
    for obj_id, mask in enumerate(initial, 1):
        predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=obj_id, mask=mask)
        cv2.imwrite(str(mask_dir / f"{obj_id:02d}_000000.png"), mask.astype(np.uint8) * 255)
    with torch.inference_mode():
        for frame_idx, obj_ids, logits in predictor.propagate_in_video(state):
            for index, obj_id in enumerate(obj_ids):
                mask = (logits[index, 0].cpu().numpy() > 0).astype(np.uint8) * 255
                cv2.imwrite(str(mask_dir / f"{int(obj_id):02d}_{frame_idx:06d}.png"), mask)
    if hasattr(predictor, "reset_state"):
        predictor.reset_state(state)
    del predictor, state
    clear_gpu()
    return {"object_count": len(initial), "mask_selection": "automatic_first_frame", "selected_masks": selection,
            "preview": str(preview_path)}


def identity_cosines(frames: list[Path], work: Path, model_id: str, device: str, objects: int) -> list[float]:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    tracks: list[list[Any]] = [[] for _ in range(objects)]
    with torch.inference_mode():
        for frame_index, path in enumerate(frames):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cannot read {path}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for obj_index in range(objects):
                mask_path = work / "masks" / f"{obj_index + 1:02d}_{frame_index:06d}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or not np.any(mask):
                    tracks[obj_index].append(None)
                    continue
                ys, xs = np.where(mask > 0)
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                crop = image[y0:y1, x0:x1].copy()
                crop[mask[y0:y1, x0:x1] == 0] = 127
                inputs = processor(images=Image.fromarray(crop), return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                vector = model(**inputs).last_hidden_state[:, 0, :]
                vector = torch.nn.functional.normalize(vector, dim=-1)
                tracks[obj_index].append(vector[0].float().cpu())
    cosines = []
    for index in range(1, len(frames)):
        values = []
        for track in tracks:
            left, right = track[index - 1], track[index]
            values.append(0.0 if left is None or right is None else float(torch.dot(left, right)))
        cosines.append(sum(values) / len(values))
    del model, processor
    clear_gpu()
    return cosines


def raft_flows(frames: list[Path], device: str) -> tuple[list[float], list[float]]:
    import cv2
    import numpy as np
    import torch
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=False).to(device).eval()
    transform = weights.transforms()
    pair_means, pair_top5 = [], []
    with torch.inference_mode():
        for first_path, second_path in zip(frames, frames[1:]):
            images = []
            for path in (first_path, second_path):
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"cannot read {path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                rgb = cv2.resize(rgb, (max(8, width // 8 * 8), max(8, height // 8 * 8)))
                images.append(torch.from_numpy(rgb).permute(2, 0, 1))
            left, right = transform(images[0], images[1])
            flow = model(left.unsqueeze(0).to(device), right.unsqueeze(0).to(device))[-1][0]
            # Equations (1) and (3) use optical-flow magnitude. Keep RAFT's
            # pixel-displacement units; diagonal normalization changes both terms.
            magnitudes = torch.linalg.vector_norm(flow, dim=0).float().cpu().numpy()
            flat = magnitudes.reshape(-1)
            top = max(1, math.ceil(flat.size * 0.05))
            pair_means.append(float(np.mean(flat)))
            pair_top5.append(float(np.mean(np.partition(flat, -top)[-top:])))
    del model
    clear_gpu()
    return pair_means, pair_top5


def depth_variance(frames: list[Path], model_id: str, device: str) -> float:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    depths = []
    with torch.inference_mode():
        for path in frames:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            raw = model(**inputs).predicted_depth[:, None]
            depth = F.interpolate(raw, size=(image.height, image.width), mode="bicubic", align_corners=False)[0, 0]
            array = depth.float().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite depth prediction for {path}")
            # Equation (2) takes variance of the estimated depth maps across
            # frames. Per-frame percentile normalization is not in the paper.
            depths.append(array)
    variance = float(np.var(np.stack(depths, axis=0), axis=0).mean())
    del model, processor
    clear_gpu()
    return variance


def temporal_ssim(frames: list[Path], flows: list[float], work: Path, args: argparse.Namespace) -> tuple[list[float], list[float]]:
    import cv2
    from skimage.metrics import structural_similarity

    pairs = work / "vfi_pairs"
    pairs.mkdir()
    mids = []
    for middle in range(1, len(frames) - 1, 2):
        left = cv2.imread(str(frames[middle - 1]), cv2.IMREAD_COLOR)
        right = cv2.imread(str(frames[middle + 1]), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise ValueError("cannot read VFI input frames")
        cv2.imwrite(str(pairs / f"{middle:06d}_left.png"), left)
        cv2.imwrite(str(pairs / f"{middle:06d}_right.png"), right)
        mids.append(middle)
    if not mids:
        raise ValueError("not enough frames for midpoint interpolation")
    predictions = work / "vfi_predictions"
    run_checked([str(args.vfi_python), str(Path(__file__).resolve()), "--vfi-worker",
                 "--repo", str(args.vfi_repo), "--pairs", str(pairs),
                 "--output", str(predictions), "--model", args.vfi_model], timeout=7200)
    similarities, weights = [], []
    for middle in mids:
        predicted = cv2.imread(str(predictions / f"{middle:06d}.png"), cv2.IMREAD_COLOR)
        actual = cv2.imread(str(frames[middle]), cv2.IMREAD_COLOR)
        if predicted is None or actual is None or predicted.shape != actual.shape:
            raise ValueError(f"missing or wrong-size VFI midpoint {middle}")
        similarities.append(float(structural_similarity(predicted, actual, data_range=255, channel_axis=2)))
        weights.append((flows[middle - 1] + flows[middle]) / 2)
    return similarities, weights


def compute_score(cosines: list[float], flow_top5: list[float], flow_mean: list[float],
                  depth_var: float, midpoint_ssim: list[float], midpoint_flow: list[float],
                  protocol: dict[str, float | int]) -> dict[str, float]:
    if not cosines or len(cosines) != len(flow_top5) or len(cosines) != len(flow_mean):
        raise ValueError("identity/flow arrays must be nonempty and aligned")
    if not midpoint_ssim or len(midpoint_ssim) != len(midpoint_flow):
        raise ValueError("midpoint SSIM/flow arrays must be nonempty and aligned")
    all_values = cosines + flow_top5 + flow_mean + [depth_var] + midpoint_ssim + midpoint_flow
    if not all(math.isfinite(float(value)) for value in all_values):
        raise ValueError("model metrics contain non-finite values")
    if depth_var < 0 or any(value < 0 for value in flow_top5 + flow_mean + midpoint_flow):
        raise ValueError("flow and depth variance must be nonnegative")
    identity = sum(cosine * min(1.0, flow / protocol["gamma"])
                   for cosine, flow in zip(cosines, flow_top5)) / len(cosines)
    depth = math.exp(-protocol["lambda_depth"] * depth_var)
    temporal = sum(ssim * math.log1p(flow)
                   for ssim, flow in zip(midpoint_ssim, midpoint_flow)) / len(midpoint_ssim)
    total = (identity + depth + temporal) / 3
    return {"S_id": identity, "S_depth": depth, "S_temp": temporal, "Score_T1": total}


def vfi_worker_main(argv: list[str]) -> int:
    """Run the VFIMamba stage in its own interpreter using this same source file."""
    parser = argparse.ArgumentParser(description="VFIMamba midpoint worker")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("VFIMamba_S", "VFIMamba"), required=True)
    args = parser.parse_args(argv)
    repo = args.repo.resolve(strict=True)
    pairs = args.pairs.resolve(strict=True)
    output = args.output.resolve()
    if not (repo / "Trainer_finetune.py").is_file():
        parser.error("--repo must contain the official VFIMamba Trainer_finetune.py")
    left_files = sorted(pairs.glob("*_left.png"))
    if not left_files:
        parser.error("--pairs contains no *_left.png files")
    output.mkdir(parents=True, exist_ok=False)

    # The upstream demo imports modules relative to the repository root.
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import cv2
    import numpy as np
    import torch
    from Trainer_finetune import Model
    from benchmark.utils.padder import InputPadder

    if not torch.cuda.is_available():
        raise RuntimeError("VFIMamba requires a CUDA GPU")
    model = Model.from_pretrained(args.model)
    model.eval()
    model.device()
    with torch.inference_mode():
        for left_path in left_files:
            stem = left_path.name[:-len("_left.png")]
            right_path = pairs / f"{stem}_right.png"
            left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
            right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
            if left is None or right is None or left.shape != right.shape:
                raise ValueError(f"missing or mismatched pair: {left_path}, {right_path}")
            first = torch.from_numpy(left.transpose(2, 0, 1)).cuda().float().unsqueeze(0) / 255
            last = torch.from_numpy(right.transpose(2, 0, 1)).cuda().float().unsqueeze(0) / 255
            padder = InputPadder(first.shape, divisor=32)
            first, last = padder.pad(first, last)
            middle = model.inference(first, last, True, TTA=False, fast_TTA=False, scale=0)
            middle = padder.unpad(middle)[0].detach().float().cpu().numpy().transpose(1, 2, 0)
            image = (np.clip(middle, 0, 1) * 255).round().astype(np.uint8)
            if not cv2.imwrite(str(output / f"{stem}.png"), image):
                raise OSError(f"could not save midpoint {stem}")
            print(f"VFI midpoint {stem}: {image.shape[1]}x{image.shape[0]}", flush=True)
    return 0


def main() -> int:
    args = arguments()
    avoid_sam2_repo_shadowing()
    try:
        protocol = load_protocol(args.protocol)
        if args.video:
            samples = standalone_specs(args.video, args.limit or len(args.video))
        else:
            rows = load_references(args.references_json, args.limit or 3)
            samples = sample_specs(rows, args)
        if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
            raise RuntimeError("ffmpeg and ffprobe are required")
        if not args.check_only:
            if not args.vfi_repo.is_dir() or not args.vfi_python.is_file():
                raise FileNotFoundError("VFI-Mamba repository or its Python interpreter is missing")
        for sample in samples:
            sample["source_duration_seconds"] = duration_seconds(sample["source"])
        if args.check_only:
            with tempfile.TemporaryDirectory(prefix="task1-preflight-") as folder:
                for sample in samples:
                    frames = extract_frames(sample, Path(folder) / sample["id"], protocol)
                    print(f"READY {sample['kind']}: {sample['video_key']} ({len(frames)} decoded frames)", flush=True)
        else:
            for sample in samples:
                print(f"READY {sample['kind']}: {sample['video_key']}", flush=True)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Preflight error: {error}", file=sys.stderr)
        return 2
    if args.check_only:
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = run_dir / "results.jsonl"
    failures = 0
    for sample in samples:
        record = {"video_key": sample["video_key"], "kind": sample["kind"], "source": sample["source"],
                  "source_duration_seconds": sample["source_duration_seconds"],
                  "protocol": protocol,
                  "protocol_source": "embedded_trial_defaults" if args.protocol is None else str(args.protocol),
                  "score_status": "experimental_paper_equations_trial_protocol",
                  "models": {"sam2": args.sam2_model, "dinov2": args.dino_model, "raft": "torchvision/Raft_Small_Weights.DEFAULT",
                             "depth": args.depth_model, "vfi": args.vfi_model},
                  "measurement_units": {"flow": "RAFT input pixels per sampled frame pair",
                                        "depth": "raw Depth Anything V2 relative depth"}}
        try:
            work = run_dir / sample["id"]
            work.mkdir()
            frames = extract_frames(sample, work / "frames", protocol)
            if "cut_seconds" in sample:
                record["cut_seconds"] = sample["cut_seconds"]
            record["sampled_frames"] = len(frames)
            record["sampled_video"] = str(frames[0].parent)
            print(f"SAM2 {sample['video_key']} ({sample['kind']})", flush=True)
            record["sam2"] = track_sam2(frames, work, args.sam2_model, args.device, protocol["max_objects"])
            print("DINOv2", flush=True)
            cosines = identity_cosines(frames, work, args.dino_model, args.device, record["sam2"]["object_count"])
            print("RAFT", flush=True)
            flows, flow_top5 = raft_flows(frames, args.device)
            print("Depth Anything V2", flush=True)
            depth_var = depth_variance(frames, args.depth_model, args.device)
            print("VFI-Mamba", flush=True)
            mid_ssim, mid_flow = temporal_ssim(frames, flows, work, args)
            record["measurements"] = {"dinov2_adjacent_cosine": cosines, "raft_flow_mean": flows,
                                      "raft_flow_top5": flow_top5, "depth_temporal_variance": depth_var,
                                      "vfi_midpoint_ssim": mid_ssim, "vfi_midpoint_flow": mid_flow}
            record["scores"] = compute_score(cosines, flow_top5, flows, depth_var, mid_ssim, mid_flow, protocol)
            record["status"] = "scored"
        except Exception as error:  # Keep a per-video diagnostic and continue the small trial.
            failures += 1
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"[:1500]
        with report.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"{record['status']}: {sample['video_key']} ({sample['kind']})", flush=True)
        if record["status"] == "error":
            print(record["error"], file=sys.stderr)
    print(f"Report: {report}")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--vfi-worker":
        raise SystemExit(vfi_worker_main(sys.argv[2:]))
    raise SystemExit(main())
