from __future__ import annotations

import argparse

import yaml

from distributed_utils import destroy_distributed, init_distributed

from .eval import run_eval_only
from .runner import train


def main():
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda, xpu, cpu")
    parser.add_argument("--overfit", type=int, default=None, help="Overfit on N images for a quick sanity check")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--epochs", type=float, default=None, help="Override total training epochs")
    parser.add_argument("--eval-only", action="store_true", help="Run validation only on a checkpoint")
    parser.add_argument("--eval-batches", type=int, default=None, help="Override validation batches; 0 means full val")
    parser.add_argument(
        "--force-random-masks",
        action="store_true",
        help="Ignore manifest masks and generate random free-form masks for the requested split.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    dist_ctx = init_distributed(args.device)
    try:
        if args.eval_only:
            run_eval_only(cfg, args, dist_ctx)
        else:
            train(cfg, args, dist_ctx)
    finally:
        destroy_distributed(dist_ctx)
