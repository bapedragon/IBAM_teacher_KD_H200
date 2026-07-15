#!/usr/bin/env python3
"""Resolve and verify the fixed ResNet56 teachers used by KD experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from train_teacher_cifar100 import CIFARResNet56


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "teachers"
DATASET_ALIASES = {
    "cifar-100": "cifar100",
    "cifar100": "cifar100",
    "flowers": "flowers102",
    "flowers-102": "flowers102",
    "flowers102": "flowers102",
    "chaoyang": "chaoyang",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT) -> dict[str, Any]:
    manifest_path = checkpoint_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Teacher manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def canonical_dataset(dataset: str) -> str:
    normalized = dataset.strip().lower()
    try:
        return DATASET_ALIASES[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(set(DATASET_ALIASES.values())))
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of: {choices}") from error


def load_teacher(
    dataset: str,
    *,
    device: str | torch.device = "cpu",
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    verify_hash: bool = True,
) -> tuple[CIFARResNet56, dict[str, Any], dict[str, Any]]:
    """Load one selected teacher, validate metadata, freeze it, and return it."""

    dataset_key = canonical_dataset(dataset)
    manifest = load_manifest(checkpoint_root)
    spec = manifest["teachers"][dataset_key]
    checkpoint_path = checkpoint_root / spec["checkpoint"]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected teacher checkpoint not found: {checkpoint_path}")
    if verify_hash:
        actual_hash = sha256(checkpoint_path)
        if actual_hash != spec["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {checkpoint_path}: "
                f"expected={spec['sha256']} actual={actual_hash}"
            )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_keys = {"model", "epoch", "accuracy", "dataset"}
    missing_keys = sorted(required_keys.difference(payload))
    if missing_keys:
        raise RuntimeError(f"Checkpoint metadata missing keys: {missing_keys}")
    if int(payload["epoch"]) != int(spec["epoch"]):
        raise RuntimeError(
            f"Epoch mismatch: manifest={spec['epoch']} checkpoint={payload['epoch']}"
        )
    if abs(float(payload["accuracy"]) - float(spec["top1"])) > 1e-8:
        raise RuntimeError(
            f"Top-1 mismatch: manifest={spec['top1']} checkpoint={payload['accuracy']}"
        )

    model = CIFARResNet56(num_classes=int(spec["num_classes"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload, spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", "cifar100", "flowers102", "chaoyang"),
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ("cifar100", "flowers102", "chaoyang") if args.dataset == "all" else (args.dataset,)
    print("=" * 72, flush=True)
    print("VERIFY FIXED TEACHER CHECKPOINTS", flush=True)
    print("=" * 72, flush=True)
    print(f"[PATH] checkpoint_root={args.checkpoint_root.resolve()}", flush=True)

    for dataset in datasets:
        model, payload, spec = load_teacher(
            dataset,
            checkpoint_root=args.checkpoint_root,
        )
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 224, 224))
        if tuple(output.shape) != (1, int(spec["num_classes"])):
            raise RuntimeError(
                f"Unexpected output shape for {dataset}: {tuple(output.shape)}"
            )
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError(f"Non-finite output detected for {dataset}")
        print(
            f"[CHECKPOINT_OK] dataset={dataset} selected={spec['selected_kind']} "
            f"epoch={payload['epoch']} top1={float(payload['accuracy']):.2f}% "
            f"classes={spec['num_classes']} sha256={spec['sha256']}",
            flush=True,
        )

    print("[DONE] All requested teacher checkpoints passed verification.", flush=True)


if __name__ == "__main__":
    main()
