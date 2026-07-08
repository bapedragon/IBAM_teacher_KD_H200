#!/usr/bin/env python3
"""Evaluate an LG/pycls DeiT-Tiny CIFAR-100 checkpoint.

This script is for checking already-trained student weights from the LG repo.
It does not train a teacher or a student. A teacher checkpoint is not required
for evaluation because distillation is only used during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.datasets.utils import check_integrity, extract_archive
from torchvision.transforms import InterpolationMode


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
REFERENCE_LG_TOP1 = 77.38
CIFAR100_SOURCES = (
    (
        "Hugging Face mirror",
        "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/"
        "201a32345d2c6b970e1a36c582930c83e09c96d2/cifar-100-python.tar.gz",
    ),
    (
        "SJTU mirror",
        "https://scidata.sjtu.edu.cn/records/xk2s3-v1e12/files/"
        "cifar-100-python.tar.gz?download=1",
    ),
    ("Toronto official", CIFAR100.url),
)


def log(message: str = "") -> None:
    print(message, flush=True)


def top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int(logits.argmax(dim=1).eq(targets).sum().item())


def cifar100_files_ready(root: Path) -> bool:
    base = root / CIFAR100.base_folder
    required_files = list(CIFAR100.train_list) + list(CIFAR100.test_list)
    required_files.append((CIFAR100.meta["filename"], CIFAR100.meta["md5"]))
    return all(check_integrity(str(base / filename), md5) for filename, md5 in required_files)


def download_cifar100_archive(url: str, destination: Path, source_name: str) -> None:
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; IBAM-H200-CIFAR100-Eval/1.0)",
            "Accept-Encoding": "identity",
        },
    )
    digest = hashlib.md5()
    downloaded = 0
    next_report_percent = 10

    log(f"[DATA] Download source={source_name} url={url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as file:
            total = int(response.headers.get("Content-Length", "0"))
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = int(downloaded * 100 / total)
                    if percent >= next_report_percent:
                        log(
                            f"[DATA] Download progress source={source_name} "
                            f"{min(percent, 100)}% ({downloaded / (1024**2):.1f} MiB)"
                        )
                        next_report_percent += 10

        actual_md5 = digest.hexdigest()
        if actual_md5 != CIFAR100.tgz_md5:
            raise RuntimeError(
                f"MD5 mismatch: expected={CIFAR100.tgz_md5} actual={actual_md5}"
            )
        partial.replace(destination)
        log(
            f"[DATA] Download verified source={source_name} "
            f"size={downloaded / (1024**2):.1f} MiB md5={actual_md5}"
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def ensure_cifar100_available(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if cifar100_files_ready(root):
        log("[DATA] Existing CIFAR-100 files passed integrity checks")
        return

    archive = root / CIFAR100.filename
    if check_integrity(str(archive), CIFAR100.tgz_md5):
        log(f"[DATA] Found verified archive; extracting {archive}")
        extract_archive(str(archive), str(root))
        if cifar100_files_ready(root):
            log("[DATA] CIFAR-100 extraction and integrity checks completed")
            return
    elif archive.exists():
        log(f"[DATA][WARN] Removing incomplete or invalid archive: {archive}")
        archive.unlink()

    failures = []
    for source_name, url in CIFAR100_SOURCES:
        for attempt in range(1, 3):
            try:
                log(f"[DATA] Attempt source={source_name} try={attempt}/2")
                download_cifar100_archive(url, archive, source_name)
                log(f"[DATA] Extracting verified archive from {source_name}")
                extract_archive(str(archive), str(root))
                if not cifar100_files_ready(root):
                    raise RuntimeError("extracted CIFAR-100 files failed integrity checks")
                log(f"[DATA] CIFAR-100 ready from {source_name}")
                return
            except Exception as error:
                message = f"{source_name} try={attempt}: {type(error).__name__}: {error}"
                failures.append(message)
                log(f"[DATA][WARN] {message}")
                archive.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(3)

    raise RuntimeError(f"All CIFAR-100 download sources failed: {' | '.join(failures)}")


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, embed_dim: int = 192) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int = 192, num_heads: int = 3) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_transform = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv_transform(x)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.projection(x)


class Mlp(nn.Module):
    def __init__(self, dim: int = 192, hidden_dim: int = 768) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        return self.fc2(x)


class Block(nn.Module):
    def __init__(self, dim: int = 192, num_heads: int = 3, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PyClsDeiTTiny(nn.Module):
    """Minimal DeiT-Tiny implementation matching LG/pycls checkpoint keys."""

    def __init__(self, image_size: int = 224, num_classes: int = 100) -> None:
        super().__init__()
        embed_dim = 192
        depth = 12
        self.patch_embed = PatchEmbed(image_size=image_size, patch_size=16, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.layers = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=3, mlp_ratio=4) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an LG/pycls DeiT-Tiny checkpoint on CIFAR-100."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-url", type=str, default="")
    parser.add_argument("--checkpoint-key", choices=("model_state", "ema_state"), default="model_state")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/eval_lg_deit_tiny"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--eval-transform",
        choices=("resize", "resize-crop"),
        default="resize",
        help=(
            "resize: directly resize CIFAR images to 224x224. "
            "resize-crop: resize to 256 then center-crop 224."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision when available.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    downloaded = 0
    next_report = 32 * 1024 * 1024
    log(f"[CKPT] Download url={url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as file:
        total = int(response.headers.get("Content-Length", "0"))
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded * 100 / total)
                if pct >= 10:
                    log(f"[CKPT] Download progress {pct}% ({downloaded / (1024**2):.1f} MiB)")
                    total = 0
                    next_report = downloaded + 32 * 1024 * 1024
            elif downloaded >= next_report:
                log(f"[CKPT] Download progress {downloaded / (1024**2):.1f} MiB")
                next_report += 32 * 1024 * 1024
    partial.replace(destination)
    log(
        f"[CKPT] Download complete path={destination.resolve()} "
        f"size={downloaded / (1024**2):.1f} MiB sha256={digest.hexdigest()}"
    )


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        return args.checkpoint
    if not args.checkpoint_url:
        raise ValueError("Pass either --checkpoint or --checkpoint-url")
    checkpoint_path = args.output_dir / "downloaded_checkpoint.pyth"
    download_file(args.checkpoint_url, checkpoint_path)
    return checkpoint_path


def make_eval_transform(image_size: int, mode: str) -> transforms.Compose:
    if mode == "resize":
        steps = [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    else:
        resize_size = int(round(image_size / 0.875))
        steps = [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    return transforms.Compose(steps)


def load_checkpoint(path: Path, key: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(payload)}")
    if key not in payload:
        raise KeyError(f"Checkpoint key '{key}' not found. Available keys: {list(payload.keys())}")
    state = payload[key]
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint['{key}'] to be a state dict, got {type(state)}")
    return payload, state


def autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader[Any], device: torch.device, amp: bool) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(amp):
            logits = model(images)
        correct += top1_correct(logits, targets)
        total += targets.numel()
    return 100.0 * correct / max(1, total)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    log("=" * 72)
    log("EVALUATE LG/PYCLS DEIT-TINY CHECKPOINT ON CIFAR-100")
    log("=" * 72)
    log(f"[ENV] python={sys.version.split()[0]} torch={torch.__version__}")
    log(f"[ENV] torchvision={torchvision.__version__}")
    log(f"[ENV] cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        log(f"[ENV] gpu_name={torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        log(f"[ENV] gpu_memory_gib={props.total_memory / (1024**3):.2f}")
    log(f"[ENV] device={device} amp={amp}")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] output_dir={args.output_dir.resolve()}")
    log(f"[EVAL] batch_size={args.batch_size} num_workers={args.num_workers}")
    log(f"[EVAL] image_size={args.image_size} eval_transform={args.eval_transform}")
    log("[NOTE] Teacher checkpoint is not needed for evaluating a trained student checkpoint.")

    checkpoint_path = resolve_checkpoint(args)
    payload, state = load_checkpoint(checkpoint_path, args.checkpoint_key, device)
    recorded_test_err = payload.get("test_err")
    recorded_top1 = None if recorded_test_err is None else 100.0 - float(recorded_test_err)
    log(f"[CKPT] path={checkpoint_path.resolve()}")
    log(f"[CKPT] keys={list(payload.keys())}")
    if recorded_top1 is not None:
        log(f"[CKPT] recorded_test_err={float(recorded_test_err):.4f} recorded_top1={recorded_top1:.2f}%")

    model = PyClsDeiTTiny(image_size=args.image_size, num_classes=100).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"[CKPT] load_state missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        log(f"[CKPT][WARN] missing_keys={missing[:20]}")
        log(f"[CKPT][WARN] unexpected_keys={unexpected[:20]}")
    if missing or unexpected:
        raise RuntimeError("Checkpoint did not exactly match PyClsDeiTTiny state_dict")

    ensure_cifar100_available(args.data_dir)
    transform = make_eval_transform(args.image_size, args.eval_transform)
    test_dataset = CIFAR100(root=args.data_dir, train=False, transform=transform, download=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    log(f"[DATA] test_samples={len(test_dataset)}")

    start = time.time()
    top1 = evaluate(model, test_loader, device, amp)
    elapsed = time.time() - start
    gap_to_reference = top1 - REFERENCE_LG_TOP1
    gap_to_recorded = None if recorded_top1 is None else top1 - recorded_top1

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_key": args.checkpoint_key,
        "recorded_top1": recorded_top1,
        "evaluated_top1": top1,
        "reference_lg_top1": REFERENCE_LG_TOP1,
        "gap_to_reference": gap_to_reference,
        "gap_to_recorded": gap_to_recorded,
        "eval_transform": args.eval_transform,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    log("=" * 72)
    log(
        f"[FINAL_RESULT] evaluated_top1={top1:.2f}% "
        f"reference_lg_top1={REFERENCE_LG_TOP1:.2f}% "
        f"gap_to_reference={gap_to_reference:+.2f}pp"
    )
    if recorded_top1 is not None:
        log(
            f"[FINAL_RESULT] recorded_top1={recorded_top1:.2f}% "
            f"gap_to_recorded={gap_to_recorded:+.2f}pp"
        )
    log(f"[FINAL_RESULT] elapsed={elapsed:.1f}s summary={summary_path.resolve()}")
    log("[DONE] Evaluation completed successfully; resources may be released.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Evaluation did not complete.")
        raise
