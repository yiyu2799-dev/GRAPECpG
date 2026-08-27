import json
import os.path as osp
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


@dataclass(frozen=True)
class SegmentSpec:
    chrom: str
    context_indices: np.ndarray
    core_start: int
    core_end: int

    @property
    def core_size(self):
        return int(self.core_end - self.core_start)


def canonical_chrom(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    text = str(value).strip()
    if text.lower().startswith('chr'):
        text = text[3:]
    text = text.strip().upper()
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
    if chrom == 'X':
        return (1, 23)
    if chrom == 'Y':
        return (1, 24)
    if chrom == 'MT':
        return (1, 25)
    return (2, chrom)


def _resolve_path(directory, filename, kind):
    path = filename if osp.isabs(filename) else osp.join(directory, filename)
    if not osp.exists(path):
        raise FileNotFoundError(f'{kind} file not found: {path}')
    return path


class MethylationGenomeDataset(Dataset):
    """Chromosome-split segment dataset shared by Hemato/Neuron-Mouse/Neuron-Homo.

    `segment_strategy="overlap"` uses a core segment plus left/right halo CpGs.
    Only core known edges are eligible as masked targets. Halo nodes/edges provide
    graph/local context and are never duplicated in the loss or metrics.
    """

    def __init__(
        self,
        processed_dir,
        split='train',
        meth_file='meth_matrix.npy',
        dna_file='dna_windows_centerC.npy',
        pos_file='pos.npy',
        chrom_file='chrom.npy',
        metadata_file='metadata.json',
        reference_lengths_file=None,
        val_chrom='5',
        test_chrom='10',
        segment_size=1024,
        segment_strategy='overlap',
        context_window=20,
        position_normalization='reference_length',
        mask_ratio=0.15,
        dynamic_train_mask=True,
        seed=1,
        max_segments=None,
        expected_dna_window=201,
        use_local=False,
        include_local_center=True,
        local_distance_scale_bp=1000.0,
    ):
        super().__init__()
        self.data_dir = str(processed_dir)
        self.split = str(split)
        self.val_chrom = canonical_chrom(val_chrom)
        self.test_chrom = canonical_chrom(test_chrom)
        self.segment_size = int(segment_size)
        self.segment_strategy = str(segment_strategy)
        self.context_window = int(context_window)
        self.position_normalization = str(position_normalization)
        self.mask_ratio = float(mask_ratio)
        self.dynamic_train_mask = bool(dynamic_train_mask)
        self.seed = int(seed)
        self.max_segments = max_segments
        self.epoch = 0
        self.use_local = bool(use_local)
        self.include_local_center = bool(include_local_center)
        self.local_distance_scale_bp = float(local_distance_scale_bp)

        if self.split not in {'train', 'val', 'test'}:
            raise ValueError('split must be train, val, or test.')
        if self.segment_size <= 0:
            raise ValueError('segment_size must be positive.')
        if self.segment_strategy not in {'overlap', 'legacy'}:
            raise ValueError('segment_strategy must be overlap or legacy.')
        if self.context_window < 0:
            raise ValueError('context_window must be non-negative.')
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError('mask_ratio must be strictly between 0 and 1.')
        if self.local_distance_scale_bp <= 0:
            raise ValueError('local_distance_scale_bp must be positive.')

        paths = {
            'meth': _resolve_path(self.data_dir, meth_file, 'methylation'),
            'dna': _resolve_path(self.data_dir, dna_file, 'DNA window'),
            'pos': _resolve_path(self.data_dir, pos_file, 'position'),
            'chrom': _resolve_path(self.data_dir, chrom_file, 'chromosome'),
        }
        for kind, path in paths.items():
            print(f'[{self.split}] {kind:5s}: {path}')

        self.meth = np.load(paths['meth'], mmap_mode='r')
        self.dna = np.load(paths['dna'], mmap_mode='r')
        self.pos = np.load(paths['pos'], mmap_mode='r')
        chrom_raw = np.load(paths['chrom'], mmap_mode='r', allow_pickle=True)
        self.chrom = np.asarray([canonical_chrom(x) for x in chrom_raw])

        self._validate_shapes(expected_dna_window)
        self.n_cells = int(self.meth.shape[1])
        self.chrom_set = sorted(set(self.chrom.tolist()), key=chrom_sort_key)
        if self.val_chrom not in self.chrom_set:
            raise ValueError(f'Validation chromosome {self.val_chrom} not found. Available: {self.chrom_set}')
        if self.test_chrom not in self.chrom_set:
            raise ValueError(f'Test chromosome {self.test_chrom} not found. Available: {self.chrom_set}')
        if self.val_chrom == self.test_chrom:
            raise ValueError('val_chrom and test_chrom must differ.')

        self.metadata = self._load_metadata(metadata_file)
        self.pos_base = int(self.metadata.get('pos_base', 0)) if self.metadata else 0
        self.chr_lengths = self._resolve_chromosome_lengths(reference_lengths_file)
        self.pos_norm = self._build_pos_norm()
        self.segments = self._build_segments()
        if max_segments is not None:
            self.segments = self.segments[: int(max_segments)]
        if not self.segments:
            raise RuntimeError(f'No segments built for split={self.split}.')

        core_sites = sum(spec.core_size for spec in self.segments)
        split_chroms = sorted({spec.chrom for spec in self.segments}, key=chrom_sort_key)
        print(
            f'[{self.split}] chromosomes={split_chroms}; segments={len(self.segments)}; '
            f'core_sites={core_sites}; cells={self.n_cells}; strategy={self.segment_strategy}; '
            f'halo={self.context_window if self.segment_strategy == "overlap" else 0}; '
            f'position_norm={self.position_normalization}'
        )

    def _validate_shapes(self, expected_dna_window):
        if self.meth.ndim != 2:
            raise ValueError(f'meth_matrix must be 2D site x cell, got {self.meth.shape}')
        if self.dna.ndim != 2:
            raise ValueError(f'dna_windows must be 2D site x window, got {self.dna.shape}')
        if self.pos.ndim != 1 or self.chrom.ndim != 1:
            raise ValueError(f'pos/chrom must be 1D, got {self.pos.shape} and {self.chrom.shape}')
        n_sites = int(self.meth.shape[0])
        if int(self.dna.shape[0]) != n_sites or len(self.pos) != n_sites or len(self.chrom) != n_sites:
            raise ValueError('Processed arrays have inconsistent site counts.')
        if expected_dna_window is not None and int(self.dna.shape[1]) != int(expected_dna_window):
            raise ValueError(
                f'DNA window mismatch: data={self.dna.shape[1]}, expected={expected_dna_window}.'
            )

    def _load_metadata(self, metadata_file):
        if metadata_file in {None, ''}:
            return {}
        path = metadata_file if osp.isabs(metadata_file) else osp.join(self.data_dir, metadata_file)
        if not osp.exists(path):
            print(f'WARNING: metadata file not found: {path}')
            return {}
        with open(path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        if metadata.get('n_cells') is not None and int(metadata['n_cells']) != int(self.meth.shape[1]):
            raise ValueError('metadata n_cells does not match meth_matrix.')
        return metadata

    @staticmethod
    def _canonicalize_length_mapping(raw):
        out = {}
        for key, value in raw.items():
            length = float(value)
            if length <= 0:
                raise ValueError(f'Invalid chromosome length for {key}: {value}')
            out[canonical_chrom(key)] = length
        return out

    def _resolve_chromosome_lengths(self, reference_lengths_file):
        if self.position_normalization == 'observed_max':
            out = {}
            for chrom in self.chrom_set:
                idx = np.where(self.chrom == chrom)[0]
                out[chrom] = max(float(np.nanmax(np.asarray(self.pos[idx], dtype=np.float64))), 1.0)
            return out

        if self.position_normalization != 'reference_length':
            raise ValueError('position_normalization must be reference_length or observed_max.')

        if reference_lengths_file:
            path = reference_lengths_file
            if not osp.isabs(path):
                path = osp.join(self.data_dir, path)
            if not osp.exists(path):
                raise FileNotFoundError(f'reference_lengths_file not found: {path}')
            with open(path, 'r', encoding='utf-8') as handle:
                raw = json.load(handle)
            lengths = self._canonicalize_length_mapping(raw)
            source = path
        else:
            raw = self.metadata.get('chromosome_lengths') if self.metadata else None
            if not raw:
                raise ValueError(
                    'reference_length normalization requires metadata.json["chromosome_lengths"] '
                    'or --reference_lengths_file. Run scripts/update_metadata_reference_lengths.py '
                    'with the original X.npz, or use --position_normalization observed_max for legacy V11 behavior.'
                )
            lengths = self._canonicalize_length_mapping(raw)
            source = 'metadata.json:chromosome_lengths'

        missing = sorted(set(self.chrom_set) - set(lengths), key=chrom_sort_key)
        if missing:
            raise ValueError(f'Missing reference chromosome lengths for: {missing}')

        # Coordinate compatibility check. `pos_base` converts source coordinates
        # to zero-based sequence indices before comparing with len(X[chrom]).
        for chrom in self.chrom_set:
            idx = np.where(self.chrom == chrom)[0]
            max_zero_based = float(np.nanmax(np.asarray(self.pos[idx], dtype=np.float64))) - self.pos_base
            if max_zero_based >= lengths[chrom]:
                raise ValueError(
                    f'Chromosome {chrom}: max zero-based CpG coordinate {max_zero_based} '
                    f'is outside reference length {lengths[chrom]} from {source}.'
                )
        return lengths

    def _build_pos_norm(self):
        pos = np.asarray(self.pos, dtype=np.float64)
        out = np.zeros_like(pos, dtype=np.float32)
        for chrom in self.chrom_set:
            idx = np.where(self.chrom == chrom)[0]
            denom = max(float(self.chr_lengths[chrom]), 1.0)
            if self.position_normalization == 'reference_length':
                coordinate = pos[idx] - float(self.pos_base)
            else:
                # Exact V11 legacy behavior: divide raw stored coordinate by max(pos).
                coordinate = pos[idx]
            out[idx] = np.clip(coordinate / denom, 0.0, 1.0).astype(np.float32)
        return out

    def _split_accepts_chrom(self, chrom):
        chrom = canonical_chrom(chrom)
        if self.split == 'val':
            return chrom == self.val_chrom
        if self.split == 'test':
            return chrom == self.test_chrom
        return chrom not in {self.val_chrom, self.test_chrom}

    def _build_segments(self):
        specs = []
        all_idx = np.arange(self.meth.shape[0], dtype=np.int64)
        halo = self.context_window if self.segment_strategy == 'overlap' else 0
        for chrom in self.chrom_set:
            if not self._split_accepts_chrom(chrom):
                continue
            chrom_idx = all_idx[self.chrom == chrom]
            chrom_idx = chrom_idx[
                np.argsort(np.asarray(self.pos[chrom_idx], dtype=np.float64), kind='mergesort')
            ]
            for core_start_global in range(0, len(chrom_idx), self.segment_size):
                core_end_global = min(core_start_global + self.segment_size, len(chrom_idx))
                context_start_global = max(0, core_start_global - halo)
                context_end_global = min(len(chrom_idx), core_end_global + halo)
                context_idx = chrom_idx[context_start_global:context_end_global].astype(np.int64)
                core_start_local = core_start_global - context_start_global
                core_end_local = core_start_local + (core_end_global - core_start_global)
                specs.append(
                    SegmentSpec(
                        chrom=chrom,
                        context_indices=context_idx,
                        core_start=int(core_start_local),
                        core_end=int(core_end_local),
                    )
                )
        return specs

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def len(self):
        return len(self.segments)

    def __len__(self):
        return len(self.segments)

    def _build_local_context(
        self, input_edge_ids, target_cells, target_sites, known_cells, known_sites, values,
        pos_seg, n_cells, n_sites, input_mat=None,
    ):
        K = int(self.context_window)
        L = 2 * K + 1
        if input_mat is None:
            input_mat = np.full((n_cells, n_sites), np.nan, dtype=np.float32)
            input_mat[known_cells[input_edge_ids], known_sites[input_edge_ids]] = values[input_edge_ids].astype(np.float32)
        else:
            if input_mat.shape != (n_cells, n_sites):
                raise ValueError(
                    f'input_mat shape mismatch: expected {(n_cells, n_sites)}, got {input_mat.shape}.'
                )
        pos_seg = pos_seg.astype(np.float64)
        denom = np.log1p(self.local_distance_scale_bp)

        site_idx_all = np.full((len(target_sites), L), -1, dtype=np.int64)
        meth_all = np.zeros((len(target_sites), L), dtype=np.int64)
        known_all = np.zeros((len(target_sites), L), dtype=np.int64)
        valid_all = np.zeros((len(target_sites), L), dtype=np.float32)
        rel_pos_idx_all = np.zeros((len(target_sites), L), dtype=np.int64)
        rel_offset_all = np.zeros((len(target_sites), L), dtype=np.int64)
        rel_dist_all = np.zeros((len(target_sites), L), dtype=np.float32)

        offsets = np.arange(-K, K + 1, dtype=np.int64)
        for row_id, (cell, site) in enumerate(zip(target_cells, target_sites)):
            cell = int(cell)
            site = int(site)
            row = input_mat[cell]
            for col_id, offset in enumerate(offsets):
                neighbor = site + int(offset)
                rel_offset_all[row_id, col_id] = int(offset)
                rel_pos_idx_all[row_id, col_id] = int(offset + K)
                if neighbor < 0 or neighbor >= n_sites:
                    continue
                if offset == 0 and not self.include_local_center:
                    continue

                site_idx_all[row_id, col_id] = neighbor
                valid_all[row_id, col_id] = 1.0
                signed_distance = float(pos_seg[neighbor]) - float(pos_seg[site])
                sign = 0.0 if signed_distance == 0 else (1.0 if signed_distance > 0 else -1.0)
                rel_dist_all[row_id, col_id] = sign * float(
                    np.log1p(abs(signed_distance)) / denom
                )
                value = row[neighbor]
                if not np.isnan(value):
                    meth_all[row_id, col_id] = int(value)
                    known_all[row_id, col_id] = 1

        return {
            'target_local_site_idx': site_idx_all,
            'target_local_meth': meth_all,
            'target_local_known': known_all,
            'target_local_valid': valid_all,
            'target_local_rel_pos_idx': rel_pos_idx_all,
            'target_local_rel_offset': rel_offset_all,
            'target_local_rel_dist': rel_dist_all,
        }

    def get(self, idx):
        spec = self.segments[idx]
        seg_idx = spec.context_indices
        chrom_values = set(self.chrom[seg_idx].tolist())
        if chrom_values != {spec.chrom}:
            raise RuntimeError(f'Segment {idx} crosses chromosomes: {chrom_values}')

        meth_seg = np.asarray(self.meth[seg_idx])
        dna_seg = np.asarray(self.dna[seg_idx])
        pos_seg = np.asarray(self.pos[seg_idx])
        pos_norm_seg = np.asarray(self.pos_norm[seg_idx], dtype=np.float32)
        if not np.all(np.diff(pos_seg.astype(np.float64)) >= 0):
            raise RuntimeError(f'Segment {idx} positions are not sorted.')

        n_sites, n_cells = int(meth_seg.shape[0]), self.n_cells
        known_sites, known_cells = np.where(meth_seg != -1)
        values = meth_seg[known_sites, known_cells].astype(np.int64)

        core_edge_mask = (known_sites >= spec.core_start) & (known_sites < spec.core_end)
        core_known_ids = np.flatnonzero(core_edge_mask)
        if len(core_known_ids) < 2:
            raise RuntimeError(
                f'Segment {idx} has only {len(core_known_ids)} known core edges; at least 2 are required.'
            )

        rng_seed = self.seed + idx
        if self.split == 'train' and self.dynamic_train_mask:
            rng_seed += self.epoch * 1000003
        rng = np.random.default_rng(rng_seed)
        n_target = max(1, int(len(core_known_ids) * self.mask_ratio))
        n_target = min(n_target, len(core_known_ids) - 1)

        if self.segment_strategy == 'legacy':
            # Exact V11 ordering for the legacy branch: randomly permute all known
            # edges, take the first n_target as targets, and keep the remaining
            # random order for graph edges. Here core == context.
            perm = rng.permutation(len(values))
            target_ids = perm[:n_target]
            input_ids = perm[n_target:]
        else:
            # Overlap strategy: only core edges may become targets. Halo edges
            # always remain available as context and are never counted in loss.
            target_perm = rng.permutation(core_known_ids)
            target_ids = target_perm[:n_target]
            is_target = np.zeros(len(values), dtype=bool)
            is_target[target_ids] = True
            input_ids = np.flatnonzero(~is_target)

        if len(input_ids) == 0:
            raise RuntimeError(f'Segment {idx} has no input edges after masking.')

        def build_bidirectional_edges(edge_ids):
            sites = known_sites[edge_ids]
            cells = known_cells[edge_ids]
            vals = values[edge_ids]
            src_fwd = cells
            dst_fwd = n_cells + sites
            src = np.concatenate([src_fwd, dst_fwd])
            dst = np.concatenate([dst_fwd, src_fwd])
            edge_index = np.stack([src, dst], axis=0)
            edge_attr = np.concatenate([vals, vals]).reshape(-1, 1)
            return edge_index, edge_attr

        edge_index, edge_attr = build_bidirectional_edges(input_ids)
        target_sites = known_sites[target_ids]
        target_cells = known_cells[target_ids]
        target_values = values[target_ids]
        target_edge_index = np.stack([target_cells, n_cells + target_sites], axis=0)

        kwargs = {
            'edge_index': torch.from_numpy(edge_index).long(),
            'edge_attr': torch.from_numpy(edge_attr).float(),
            'target_edge_index': torch.from_numpy(target_edge_index).long(),
            'target_labels': torch.from_numpy(target_values).float(),
            'dna_seg': torch.from_numpy(dna_seg).long(),
            'pos_seg': torch.from_numpy(pos_seg).long(),
            'pos_norm_seg': torch.from_numpy(pos_norm_seg).float(),
            'n_cells': torch.tensor([n_cells], dtype=torch.long),
            'n_sites': torch.tensor([n_sites], dtype=torch.long),
        }

        if self.use_local:
            local = self._build_local_context(
                input_edge_ids=input_ids,
                target_cells=target_cells,
                target_sites=target_sites,
                known_cells=known_cells,
                known_sites=known_sites,
                values=values,
                pos_seg=pos_seg,
                n_cells=n_cells,
                n_sites=n_sites,
            )
            kwargs.update({
                'target_local_site_idx': torch.from_numpy(local['target_local_site_idx']).long(),
                'target_local_meth': torch.from_numpy(local['target_local_meth']).long(),
                'target_local_known': torch.from_numpy(local['target_local_known']).long(),
                'target_local_valid': torch.from_numpy(local['target_local_valid']).float(),
                'target_local_rel_pos_idx': torch.from_numpy(local['target_local_rel_pos_idx']).long(),
                'target_local_rel_offset': torch.from_numpy(local['target_local_rel_offset']).long(),
                'target_local_rel_dist': torch.from_numpy(local['target_local_rel_dist']).float(),
            })

        data = Data(**kwargs)
        data.num_nodes = int(n_cells + n_sites)
        return data
