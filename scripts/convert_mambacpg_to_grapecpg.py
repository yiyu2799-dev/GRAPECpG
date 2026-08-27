#!/usr/bin/env python3
"""Convert MambaCpG-style X/y/pos NPZ files into generic GRAPE-CpG arrays."""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


def canonical_chrom(value):
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


def chrom_sort_key(chrom):
    chrom = canonical_chrom(chrom)
    if chrom.isdigit():
        return (0, int(chrom))
    return (1, {'X': 23, 'Y': 24, 'MT': 25}.get(chrom, 99), chrom)


def clean_dna(sequence):
    sequence = np.asarray(sequence)
    sequence = np.where((sequence >= 0) & (sequence <= 3), sequence, 4)
    return sequence.astype(np.int8)


def infer_pos_base(sequence, positions, max_samples=200000):
    sequence = clean_dna(sequence)
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) > max_samples:
        positions = positions[np.linspace(0, len(positions) - 1, max_samples).astype(np.int64)]
    scores = {}
    for base in (0, 1):
        centers = positions - base
        valid = (centers >= 0) & (centers + 1 < len(sequence))
        centers = centers[valid]
        if len(centers) == 0:
            scores[base] = 0.0
            continue
        pairs = sequence[centers].astype(np.int16) * 16 + sequence[centers + 1].astype(np.int16)
        _, counts = np.unique(pairs, return_counts=True)
        scores[base] = float(counts.max() / len(pairs))
    return (0 if scores[0] >= scores[1] else 1), scores


def validate_methylation_sample(matrix, chrom, sample_rows=4096):
    n_rows = matrix.shape[0]
    idx = np.linspace(0, n_rows - 1, min(sample_rows, n_rows), dtype=np.int64)
    values = set(np.unique(np.asarray(matrix[idx])).tolist())
    if not values.issubset({-1, 0, 1}):
        raise ValueError(f'{chrom} has unexpected methylation values: {sorted(values)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--x_file', required=True)
    parser.add_argument('--y_file', required=True)
    parser.add_argument('--pos_file', required=True)
    parser.add_argument('--dataset_name', default='dataset')
    parser.add_argument('--window', type=int, default=201)
    parser.add_argument('--pos_base', default='auto', choices=['auto', '0', '1'])
    parser.add_argument('--block_size', type=int, default=50000)
    parser.add_argument('--exclude_chroms', default='')
    parser.add_argument('--expected_cells', type=int, default=None)
    parser.add_argument('--val_chrom', default='5')
    parser.add_argument('--test_chrom', default='10')
    args = parser.parse_args()

    if args.window <= 0 or args.window % 2 != 1:
        raise ValueError('--window must be a positive odd integer.')
    if args.block_size <= 0:
        raise ValueError('--block_size must be positive.')

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path, p_path = input_dir / args.x_file, input_dir / args.y_file, input_dir / args.pos_file
    for path in (x_path, y_path, p_path):
        if not path.exists():
            raise FileNotFoundError(path)

    X = np.load(x_path, allow_pickle=True)
    Y = np.load(y_path, allow_pickle=True)
    P = np.load(p_path, allow_pickle=True)
    common = set(X.files) & set(Y.files) & set(P.files)
    if not common:
        raise ValueError('X/y/pos NPZ files have no common chromosome keys.')
    chromosomes = sorted(common, key=chrom_sort_key)
    if args.exclude_chroms.strip():
        excluded = {canonical_chrom(v) for v in args.exclude_chroms.split(',') if v.strip()}
        chromosomes = [c for c in chromosomes if canonical_chrom(c) not in excluded]

    n_cells = None
    total_sites = 0
    chrom_counts = {}
    chromosome_lengths = {}
    for chrom in chromosomes:
        y = Y[chrom]
        pos = P[chrom]
        if y.ndim != 2 or pos.ndim != 1 or y.shape[0] != pos.shape[0]:
            raise ValueError(f'{chrom}: incompatible y/pos shapes: {y.shape}, {pos.shape}')
        validate_methylation_sample(y, chrom)
        n_cells = int(y.shape[1]) if n_cells is None else n_cells
        if int(y.shape[1]) != n_cells:
            raise ValueError(f'{chrom}: cell count differs from previous chromosomes.')
        chrom_counts[canonical_chrom(chrom)] = int(y.shape[0])
        chromosome_lengths[canonical_chrom(chrom)] = int(len(X[chrom]))
        total_sites += int(y.shape[0])
        print(f'{chrom}: sites={y.shape[0]} cells={y.shape[1]} reference_length={len(X[chrom])}')

    if args.expected_cells is not None and n_cells != args.expected_cells:
        raise ValueError(f'Expected {args.expected_cells} cells but source has {n_cells}.')

    if args.pos_base == 'auto':
        bases, details = [], {}
        for chrom in chromosomes[: min(3, len(chromosomes))]:
            base, scores = infer_pos_base(X[chrom], P[chrom])
            bases.append(base)
            details[canonical_chrom(chrom)] = scores
            print(f'{chrom}: position-base scores={scores}; chosen={base}')
        pos_base = int(round(float(np.mean(bases)))) if bases else 0
    else:
        pos_base = int(args.pos_base)
        details = None
    print('pos_base =', pos_base)

    # Validate that positions and X use a compatible coordinate system.
    for chrom in chromosomes:
        zero_based = np.asarray(P[chrom], dtype=np.int64) - pos_base
        if zero_based.size and (zero_based.min() < 0 or zero_based.max() >= len(X[chrom])):
            raise ValueError(
                f'{chrom}: CpG positions fall outside X sequence after pos_base={pos_base}; '
                f'range={zero_based.min()}..{zero_based.max()}, len(X)={len(X[chrom])}'
            )

    meth = open_memmap(output_dir / 'meth_matrix.npy', mode='w+', dtype=np.int8, shape=(total_sites, n_cells))
    dna = open_memmap(
        output_dir / 'dna_windows_centerC.npy', mode='w+', dtype=np.int8, shape=(total_sites, args.window)
    )
    pos_out = open_memmap(output_dir / 'pos.npy', mode='w+', dtype=np.int32, shape=(total_sites,))
    chrom_width = max(len(canonical_chrom(c)) for c in chromosomes)
    chrom_out = open_memmap(
        output_dir / 'chrom.npy', mode='w+', dtype=f'<U{chrom_width}', shape=(total_sites,)
    )

    half = args.window // 2
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    output_offset = 0
    known_summary = {}
    for chrom in chromosomes:
        y = np.asarray(Y[chrom], dtype=np.int8)
        positions = np.asarray(P[chrom], dtype=np.int32)
        sequence = clean_dna(X[chrom])
        n_sites = int(y.shape[0])
        start, end = output_offset, output_offset + n_sites
        meth[start:end] = y
        pos_out[start:end] = positions
        chrom_out[start:end] = canonical_chrom(chrom)

        centers = positions.astype(np.int64) - pos_base
        for block_start in range(0, n_sites, args.block_size):
            block_end = min(block_start + args.block_size, n_sites)
            index = centers[block_start:block_end, None] + offsets[None, :]
            valid = (index >= 0) & (index < len(sequence))
            clipped = np.clip(index, 0, len(sequence) - 1)
            windows = sequence[clipped]
            windows[~valid] = 4
            dna[start + block_start:start + block_end] = windows.astype(np.int8)

        known = int((y != -1).sum())
        ones = int((y == 1).sum())
        known_summary[canonical_chrom(chrom)] = {
            'sites': n_sites,
            'known_edges': known,
            'known_ratio': float(known / y.size),
            'one_ratio_among_known': float(ones / known) if known else None,
        }
        meth.flush(); dna.flush(); pos_out.flush(); chrom_out.flush()
        output_offset = end

    metadata = {
        'dataset_name': args.dataset_name,
        'source_files': {'X': str(x_path), 'y': str(y_path), 'pos': str(p_path)},
        'chromosomes': [canonical_chrom(c) for c in chromosomes],
        'chrom_counts': chrom_counts,
        'chromosome_lengths': chromosome_lengths,
        'chromosome_length_source': 'len(X[chrom]) from the supplied MambaCpG-style reference sequence NPZ',
        'reference_assembly': None,
        'total_sites': total_sites,
        'n_cells': n_cells,
        'window': args.window,
        'pos_base': pos_base,
        'pos_base_detail': details,
        'known_summary': known_summary,
        'split_rule': {
            'val': canonical_chrom(args.val_chrom),
            'test': canonical_chrom(args.test_chrom),
            'train': 'all other chromosomes',
        },
        'dna_note': 'DNA tokens outside 0/1/2/3 are converted to 4 (unknown/pad).',
    }
    with (output_dir / 'metadata.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    print('Conversion finished:', output_dir)


if __name__ == '__main__':
    main()
