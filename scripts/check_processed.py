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
    for split_name, split_chrom in [('validation', args.val_chrom), ('test', args.test_chrom)]:
        split_chrom = canonical_chrom(split_chrom)
        if split_chrom not in chroms:
            raise ValueError(f'{split_name} chromosome {split_chrom} not found. Available={chroms}')

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
