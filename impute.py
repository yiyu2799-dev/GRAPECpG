#!/usr/bin/env python3
"""Impute originally unknown methylation entries with a trained GRAPE-CpG model.

Predictions are written as per-segment NPZ shards instead of a dense genome-wide
matrix, because large single-cell methylation datasets can contain billions of
unknown cell-CpG pairs.
"""

import argparse
import copy
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from models.model import GrapeCpGModel
from training.checkpoints import load_raw_checkpoint, load_strict
from training.trainer import build_dataset


def _checkpoint_args(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get('config'), dict):
        raise ValueError('Imputation requires a checkpoint with an embedded run config.')
    return Namespace(**dict(raw['config']))


def _dataset_from_checkpoint(args, split, processed_dir, max_segments):
    # Reuse the same split-mode router as training/evaluation.  Work on a copy
    # so command-line path overrides do not mutate the embedded checkpoint config.
    dataset_args = copy.copy(args)
    dataset_args.processed_dir = processed_dir
    dataset_args.dynamic_train_mask = False
    return build_dataset(dataset_args, split, int(dataset_args.seed), max_segments)


def _full_observed_graph(dataset, spec):
    seg_idx = spec.context_indices
    meth_seg = np.asarray(dataset.meth[seg_idx])
    dna_seg = np.asarray(dataset.dna[seg_idx])
    pos_seg = np.asarray(dataset.pos[seg_idx])
    pos_norm_seg = np.asarray(dataset.pos_norm[seg_idx], dtype=np.float32)
    n_sites, n_cells = int(meth_seg.shape[0]), int(dataset.n_cells)

    known_sites, known_cells = np.where(meth_seg != -1)
    values = meth_seg[known_sites, known_cells].astype(np.int64)
    if len(values) == 0:
        raise RuntimeError(f'Chromosome {spec.chrom} segment has no observed methylation edges.')

    src_fwd = known_cells
    dst_fwd = n_cells + known_sites
    edge_index = np.stack(
        [np.concatenate([src_fwd, dst_fwd]), np.concatenate([dst_fwd, src_fwd])],
        axis=0,
    )
    edge_attr = np.concatenate([values, values]).reshape(-1, 1)
    data = Data(
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        dna_seg=torch.from_numpy(dna_seg).long(),
        pos_seg=torch.from_numpy(pos_seg).long(),
        pos_norm_seg=torch.from_numpy(pos_norm_seg).float(),
        n_cells=torch.tensor([n_cells], dtype=torch.long),
        n_sites=torch.tensor([n_sites], dtype=torch.long),
    )
    return data, meth_seg, pos_seg, known_sites, known_cells, values


def _attach_targets(dataset, data, target_cells, target_sites, pos_seg, known_sites, known_cells, values, input_mat=None):
    n_cells = int(dataset.n_cells)
    n_sites = int(data.n_sites.item())
    target_edge_index = np.stack([target_cells, n_cells + target_sites], axis=0)
    data.target_edge_index = torch.from_numpy(target_edge_index).long()

    if dataset.use_local:
        all_known_ids = np.arange(len(values), dtype=np.int64)
        local = dataset._build_local_context(
            input_edge_ids=all_known_ids,
            target_cells=target_cells,
            target_sites=target_sites,
            known_cells=known_cells,
            known_sites=known_sites,
            values=values,
            pos_seg=pos_seg,
            n_cells=n_cells,
            n_sites=n_sites,
            input_mat=input_mat,
        )
        data.target_local_site_idx = torch.from_numpy(local['target_local_site_idx']).long()
        data.target_local_meth = torch.from_numpy(local['target_local_meth']).long()
        data.target_local_known = torch.from_numpy(local['target_local_known']).long()
        data.target_local_valid = torch.from_numpy(local['target_local_valid']).float()
        data.target_local_rel_pos_idx = torch.from_numpy(local['target_local_rel_pos_idx']).long()
        data.target_local_rel_offset = torch.from_numpy(local['target_local_rel_offset']).long()
        data.target_local_rel_dist = torch.from_numpy(local['target_local_rel_dist']).float()
    return data


def _process_split(model, dataset, device, output_dir, chunk_size, split_name):
    manifest = []
    for segment_id, spec in enumerate(dataset.segments):
        graph, meth_seg, pos_seg, known_sites, known_cells, values = _full_observed_graph(dataset, spec)
        core = meth_seg[spec.core_start:spec.core_end]
        core_site_offset, target_cells_all = np.where(core == -1)
        target_sites_all = core_site_offset.astype(np.int64) + int(spec.core_start)
        target_cells_all = target_cells_all.astype(np.int64)
        if len(target_sites_all) == 0:
            continue

        graph = graph.to(device)
        with torch.no_grad():
            node_emb = model.encode_global(graph)

        input_mat = None
        if dataset.use_local:
            input_mat = np.full((int(dataset.n_cells), int(meth_seg.shape[0])), np.nan, dtype=np.float32)
            input_mat[known_cells, known_sites] = values.astype(np.float32)

        shard_global_site = []
        shard_cell = []
        shard_prob = []
        for start in range(0, len(target_sites_all), chunk_size):
            end = min(start + chunk_size, len(target_sites_all))
            target_sites = target_sites_all[start:end]
            target_cells = target_cells_all[start:end]
            target_data = Data(
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                dna_seg=graph.dna_seg,
                pos_seg=graph.pos_seg,
                pos_norm_seg=graph.pos_norm_seg,
                n_cells=graph.n_cells,
                n_sites=graph.n_sites,
            )
            target_data = _attach_targets(
                dataset,
                target_data,
                target_cells,
                target_sites,
                pos_seg,
                known_sites,
                known_cells,
                values,
                input_mat=input_mat,
            ).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model.decode_targets(node_emb, target_data)).cpu().numpy().astype(np.float32)
            shard_global_site.append(spec.context_indices[target_sites].astype(np.int64))
            shard_cell.append(target_cells.astype(np.int32))
            shard_prob.append(prob)

        global_site = np.concatenate(shard_global_site)
        cell_index = np.concatenate(shard_cell)
        probability = np.concatenate(shard_prob)
        filename = f'{split_name}_chr{spec.chrom}_segment{segment_id:06d}.npz'
        path = output_dir / filename
        np.savez(
            path,
            global_site_index=global_site,
            cell_index=cell_index,
            probability=probability,
            predicted_state=(probability >= 0.5).astype(np.int8),
        )
        manifest.append(
            {
                'file': filename,
                'split': split_name,
                'chrom': spec.chrom,
                'segment_id': segment_id,
                'n_predictions': int(len(probability)),
            }
        )
        print(f'[{split_name}] chr{spec.chrom} segment={segment_id}: {len(probability)} predictions -> {path}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Impute unknown CpG methylation states with GRAPE-CpG.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--processed_dir', default=None,
                        help='Override the processed-data path embedded in the checkpoint.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--max_segments', type=int, default=None)
    parser.add_argument('--target_chunk_size', type=int, default=8192)
    args_cli = parser.parse_args()
    if args_cli.target_chunk_size <= 0:
        raise ValueError('--target_chunk_size must be positive.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw, _ = load_raw_checkpoint(args_cli.checkpoint, device)
    args = _checkpoint_args(raw)
    processed_dir = args_cli.processed_dir or args.processed_dir
    if not processed_dir:
        raise ValueError('processed_dir is missing from both the checkpoint and command line.')

    model = GrapeCpGModel(args).to(device)
    load_strict(model, args_cli.checkpoint, device)
    model.eval()

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ['train', 'val', 'test'] if args_cli.split == 'all' else [args_cli.split]
    manifest = []
    for split in splits:
        dataset = _dataset_from_checkpoint(args, split, processed_dir, args_cli.max_segments)
        if int(dataset.n_cells) != int(args.n_cells):
            raise ValueError(
                f'Checkpoint expects {args.n_cells} cells but processed data contains {dataset.n_cells}.'
            )
        manifest.extend(
            _process_split(
                model,
                dataset,
                device,
                output_dir,
                args_cli.target_chunk_size,
                split,
            )
        )

    (output_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'checkpoint': str(Path(args_cli.checkpoint).resolve()),
                'processed_dir': str(Path(processed_dir).resolve()),
                'threshold': 0.5,
                'shards': manifest,
                'total_predictions': int(sum(x['n_predictions'] for x in manifest)),
            },
            indent=2,
            sort_keys=True,
        ) + '\n',
        encoding='utf-8',
    )
    print('Imputation finished. Manifest:', output_dir / 'manifest.json')


if __name__ == '__main__':
    main()
