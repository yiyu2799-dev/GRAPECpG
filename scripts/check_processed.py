#!/usr/bin/env python3
"""Fast preflight checks for generic processed GRAPE-CpG arrays."""
import argparse
import json
from pathlib import Path

import numpy as np


def canonical_chrom(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    text = str(value).strip()
    if text.lower().startswith('chr'):
        text = text[3:]
    text = text.upper()
    if text in {'M', 'MT'}:
        return 'MT'
    if text in {'X', 'Y'}:
        return text
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', required=True)
    parser.add_argument('--meth_file', default='meth_matrix.npy')
    parser.add_argument('--dna_file', default='dna_windows_centerC.npy')
    parser.add_argument('--pos_file', default='pos.npy')
    parser.add_argument('--chrom_file', default='chrom.npy')
    parser.add_argument('--metadata_file', default='metadata.json')
    parser.add_argument('--expected_cells', type=int, default=None)
    parser.add_argument('--expected_window', type=int, default=None)
    parser.add_argument(
        '--split_mode', choices=['chromosome_holdout', 'within_chromosome'],
        default='chromosome_holdout',
    )
    parser.add_argument('--split_chrom', default=None)
    parser.add_argument(
        '--split_fractions', nargs=3, type=float, default=None,
        metavar=('TRAIN', 'VAL', 'TEST'),
    )
    parser.add_argument('--val_chrom', default='5')
    parser.add_argument('--test_chrom', default='10')
    parser.add_argument('--sample_sites', type=int, default=4096)
    parser.add_argument('--require_reference_lengths', action='store_true')
    args = parser.parse_args()

    directory = Path(args.processed_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    paths = {
        'meth': directory / args.meth_file,
        'dna': directory / args.dna_file,
        'pos': directory / args.pos_file,
        'chrom': directory / args.chrom_file,
    }
    for kind, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f'{kind} file not found: {path}')
        print(f'{kind:5s}: {path.name} ({path.stat().st_size / (1024.0 ** 3):.3f} GiB)')

    meth = np.load(paths['meth'], mmap_mode='r')
    dna = np.load(paths['dna'], mmap_mode='r')
    pos = np.load(paths['pos'], mmap_mode='r')
    chrom_raw = np.load(paths['chrom'], mmap_mode='r', allow_pickle=True)
    chrom = np.asarray([canonical_chrom(x) for x in chrom_raw], dtype=object)

    if meth.ndim != 2 or dna.ndim != 2 or pos.ndim != 1 or chrom.ndim != 1:
        raise ValueError(f'Unexpected dimensions: meth={meth.shape}, dna={dna.shape}, pos={pos.shape}, chrom={chrom.shape}')
    n_sites, n_cells = int(meth.shape[0]), int(meth.shape[1])
    if int(dna.shape[0]) != n_sites or len(pos) != n_sites or len(chrom) != n_sites:
        raise ValueError('Processed arrays have inconsistent site counts.')
    if args.expected_cells is not None and n_cells != args.expected_cells:
        raise ValueError(f'Expected {args.expected_cells} cells, got {n_cells}.')
    if args.expected_window is not None and int(dna.shape[1]) != args.expected_window:
        raise ValueError(f'Expected DNA window {args.expected_window}, got {dna.shape[1]}.')

    metadata_path = directory / args.metadata_file
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open('r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        if metadata.get('n_cells') is not None and int(metadata['n_cells']) != n_cells:
            raise ValueError('metadata n_cells does not match meth_matrix.')
        if metadata.get('window') is not None and int(metadata['window']) != int(dna.shape[1]):
            raise ValueError('metadata window does not match dna array.')
    else:
        print('WARNING: metadata.json is missing.')

    chroms = sorted(set(chrom.tolist()))
    if args.split_mode == 'chromosome_holdout':
        if args.split_chrom is not None or args.split_fractions is not None:
            raise ValueError(
                '--split_chrom/--split_fractions are only valid with '
                '--split_mode within_chromosome.'
            )
        for split_name, split_chrom in [('validation', args.val_chrom), ('test', args.test_chrom)]:
            split_chrom = canonical_chrom(split_chrom)
            if split_chrom not in chroms:
                raise ValueError(f'{split_name} chromosome {split_chrom} not found. Available={chroms}')
        print(
            'Split check: chromosome_holdout PASSED '
            f'(val={canonical_chrom(args.val_chrom)}, test={canonical_chrom(args.test_chrom)})'
        )
    else:
        if args.split_chrom in {None, ''}:
            raise ValueError('--split_chrom is required with --split_mode within_chromosome.')
        if args.split_fractions is None:
            raise ValueError('--split_fractions TRAIN VAL TEST is required with --split_mode within_chromosome.')

        split_chrom = canonical_chrom(args.split_chrom)
        fractions = tuple(float(x) for x in args.split_fractions)
        if len(fractions) != 3 or not all(np.isfinite(x) and x > 0 for x in fractions):
            raise ValueError('split_fractions must be three finite, strictly positive values.')
        if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError(f'split_fractions must sum to 1.0, got {fractions}.')
        if chroms != [split_chrom]:
            raise ValueError(
                'within_chromosome mode requires exactly the requested chromosome; '
                f'requested={split_chrom}, available={chroms}.'
            )

        idx = np.flatnonzero(chrom == split_chrom).astype(np.int64)
        pos_selected = np.asarray(pos[idx], dtype=np.float64)
        order = np.argsort(pos_selected, kind='mergesort')
        sorted_idx = idx[order]
        sorted_pos = pos_selected[order]
        if len(sorted_pos) > 1 and np.any(np.diff(sorted_pos) <= 0):
            raise ValueError(
                'within_chromosome split requires unique genomic positions; '
                'duplicate/non-increasing positions were detected.'
            )

        train_end = int(np.floor(len(sorted_idx) * fractions[0]))
        val_end = int(np.floor(len(sorted_idx) * (fractions[0] + fractions[1])))
        split_indices = {
            'train': sorted_idx[:train_end],
            'val': sorted_idx[train_end:val_end],
            'test': sorted_idx[val_end:],
        }
        counts = {name: int(len(x)) for name, x in split_indices.items()}
        if any(count <= 0 for count in counts.values()):
            raise ValueError(
                f'One or more within-chromosome splits are empty: {counts}.'
            )

        concatenated = np.concatenate([split_indices['train'], split_indices['val'], split_indices['test']])
        if len(np.unique(concatenated)) != len(sorted_idx):
            raise RuntimeError('Split overlap detected.')
        if not np.array_equal(concatenated, sorted_idx):
            raise RuntimeError('Split union/order invariant failed.')

        train_pos = np.asarray(pos[split_indices['train']], dtype=np.float64)
        val_pos = np.asarray(pos[split_indices['val']], dtype=np.float64)
        test_pos = np.asarray(pos[split_indices['test']], dtype=np.float64)
        if not (train_pos[-1] < val_pos[0] and val_pos[-1] < test_pos[0]):
            raise RuntimeError('Position boundaries are not strictly ordered train < val < test.')
        print(
            'Split check: within_chromosome PASSED '
            f'(chr={split_chrom}, fractions={fractions}, counts={counts}, '
            f'boundaries={train_pos[-1]}|{val_pos[0]} ... {val_pos[-1]}|{test_pos[0]})'
        )

    sample_n = min(max(args.sample_sites, 1), n_sites)
    sample_idx = np.linspace(0, n_sites - 1, sample_n, dtype=np.int64)
    meth_values = set(np.unique(np.asarray(meth[sample_idx])).tolist())
    dna_values = set(np.unique(np.asarray(dna[sample_idx])).tolist())
    if not meth_values.issubset({-1, 0, 1}):
        raise ValueError(f'Unexpected methylation values: {sorted(meth_values)}')
    if not dna_values.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f'Unexpected DNA tokens: {sorted(dna_values)}')

    lengths = metadata.get('chromosome_lengths') if metadata else None
    if args.require_reference_lengths and not lengths:
        raise ValueError(
            'Reference chromosome lengths are required but absent from metadata. '
            'Run scripts/update_metadata_reference_lengths.py with the original X.npz.'
        )
    if lengths:
        lengths = {canonical_chrom(k): int(v) for k, v in lengths.items()}
        pos_base = int(metadata.get('pos_base', 0))
        for c in chroms:
            if c not in lengths:
                raise ValueError(f'Missing reference length for chromosome {c}.')
            idx = np.where(chrom == c)[0]
            max_zero_based = int(np.max(np.asarray(pos[idx], dtype=np.int64)) - pos_base)
            if max_zero_based >= lengths[c]:
                raise ValueError(
                    f'Chromosome {c} coordinate/reference mismatch: max pos={max_zero_based}, length={lengths[c]}.'
                )
        print('Reference chromosome length check: PASSED')

    known = int((np.asarray(meth[sample_idx]) != -1).sum())
    ones = int((np.asarray(meth[sample_idx]) == 1).sum())
    print('Shape/value checks: PASSED')
    print(f'sites={n_sites}, cells={n_cells}, DNA window={dna.shape[1]}, chromosomes={chroms}')
    print(f'sampled known ratio={known / float(sample_n * n_cells):.6f}, methylation rate={ones / float(max(known, 1)):.6f}')


if __name__ == '__main__':
    main()
