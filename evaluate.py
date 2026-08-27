#!/usr/bin/env python3
"""Evaluate a saved GRAPE-CpG checkpoint without retraining.

The model/data configuration is restored from the checkpoint itself to avoid
silent architecture mismatches.  Only data location and evaluation scope may
be overridden from the command line.
"""

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from models.model import GrapeCpGModel
from training.checkpoints import load_raw_checkpoint, load_strict
from training.trainer import build_dataset, build_loader_kwargs, evaluate_model


def _checkpoint_args(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get('config'), dict):
        raise ValueError(
            'This checkpoint has no embedded config. Use a checkpoint produced by the cleaned '
            'training code, or evaluate it through the original run configuration.'
        )
    return Namespace(**dict(raw['config']))


def main():
    parser = argparse.ArgumentParser(description='Evaluate a GRAPE-CpG checkpoint.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--processed_dir', default=None,
                        help='Override the processed-data directory stored in the checkpoint.')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--max_segments', type=int, default=None)
    parser.add_argument('--output', default=None, help='Optional JSON output path.')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', action='store_true', default=None)
    args_cli = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw, _ = load_raw_checkpoint(args_cli.checkpoint, device)
    args = _checkpoint_args(raw)

    if args_cli.processed_dir is not None:
        args.processed_dir = args_cli.processed_dir
    if args_cli.num_workers is not None:
        args.num_workers = args_cli.num_workers
    if args_cli.pin_memory is not None:
        args.pin_memory = args_cli.pin_memory

    # Older compatible checkpoints may predate these runtime-only fields.
    if not hasattr(args, 'num_workers'):
        args.num_workers = 0
    if not hasattr(args, 'prefetch_factor'):
        args.prefetch_factor = 2
    if not hasattr(args, 'pin_memory'):
        args.pin_memory = False

    seed_offset = 100000 if args_cli.split == 'val' else 200000
    default_limit = getattr(args, 'max_val_segments' if args_cli.split == 'val' else 'max_test_segments', None)
    max_segments = args_cli.max_segments if args_cli.max_segments is not None else default_limit
    dataset = build_dataset(args, args_cli.split, int(args.seed) + seed_offset, max_segments)
    if int(dataset.n_cells) != int(args.n_cells):
        raise ValueError(
            f'Checkpoint expects {args.n_cells} cells but processed data contains {dataset.n_cells}.'
        )

    loader_kwargs = build_loader_kwargs(args)
    loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
    model = GrapeCpGModel(args).to(device)
    load_strict(model, args_cli.checkpoint, device)
    metrics = evaluate_model(
        model,
        loader,
        device,
        args,
        non_blocking=bool(args.pin_memory and device.type == 'cuda'),
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args_cli.output:
        output = Path(args_cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
