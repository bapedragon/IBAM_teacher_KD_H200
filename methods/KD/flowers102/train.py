#!/usr/bin/env python3
"""Run standard logit KD on Oxford Flowers 102."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.KD.core import cli_main


if __name__ == "__main__":
    if any(arg == "--dataset" or arg.startswith("--dataset=") for arg in sys.argv[1:]):
        raise SystemExit("This wrapper fixes --dataset flowers102; remove --dataset from the command.")
    sys.argv[1:1] = ["--dataset", "flowers102"]
    cli_main()
