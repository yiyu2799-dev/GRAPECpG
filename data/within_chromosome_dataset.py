"""Single-chromosome position-split dataset for GRAPE-CpG.

The existing chromosome-holdout dataset is intentionally left untouched.
This class adds an independent channel in which one chromosome is sorted by
position and split into contiguous train/validation/test regions by CpG count.
Segment halo/context is built *after* the split, so context can never cross a
train/validation/test boundary.
"""

import numpy as np
from torch_geometric.data import Dataset

from data.dataset import (
    MethylationGenomeDataset,
    SegmentSpec,
    _resolve_path,
    canonical_chrom,
    chrom_sort_key,
)
from data.splitters import split_indices_by_position, validate_split_fractions


class WithinChromosomeMethylationDataset(MethylationGenomeDataset):
    """Contiguous position split within one chromosome.

    Common graph construction, masking, local-context construction, and sample
    materialization are inherited from :class:`MethylationGenomeDataset`.
    Only initialization and segment construction differ from the legacy
    chromosome-holdout channel.
    """

    def __init__(
        self,
        processed_dir,
        split="train",
        meth_file="meth_matrix.npy",
        dna_file="dna_windows_centerC.npy",
        pos_file="pos.npy",
        chrom_file="chrom.npy",
        metadata_file="metadata.json",
        reference_lengths_file=None,
        split_chrom=None,
        split_fractions=None,
        segment_size=1024,
        segment_strategy="overlap",
        context_window=20,
        position_normalization="reference_length",
        mask_ratio=0.15,
        dynamic_train_mask=True,
        seed=1,
        max_segments=None,
        expected_dna_window=201,
        use_local=False,
        include_local_center=True,
        local_distance_scale_bp=1000.0,
    ):
        # Do not call MethylationGenomeDataset.__init__: that constructor
        # intentionally enforces the legacy val_chrom/test_chrom protocol.
        Dataset.__init__(self)

        self.data_dir = str(processed_dir)
        self.split = str(split)
        self.split_chrom = canonical_chrom(split_chrom) if split_chrom not in {None, ""} else None
        self.split_fractions = validate_split_fractions(split_fractions)
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

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test.")
        if self.split_chrom is None:
            raise ValueError("split_chrom must be provided for within_chromosome mode.")
        if self.segment_size <= 0:
            raise ValueError("segment_size must be positive.")
        if self.segment_strategy not in {"overlap", "legacy"}:
            raise ValueError("segment_strategy must be overlap or legacy.")
        if self.context_window < 0:
            raise ValueError("context_window must be non-negative.")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be strictly between 0 and 1.")
        if self.local_distance_scale_bp <= 0:
            raise ValueError("local_distance_scale_bp must be positive.")

        paths = {
            "meth": _resolve_path(self.data_dir, meth_file, "methylation"),
            "dna": _resolve_path(self.data_dir, dna_file, "DNA window"),
            "pos": _resolve_path(self.data_dir, pos_file, "position"),
            "chrom": _resolve_path(self.data_dir, chrom_file, "chromosome"),
        }
        for kind, path in paths.items():
            print(f"[{self.split}] {kind:5s}: {path}")

        self.meth = np.load(paths["meth"], mmap_mode="r")
        self.dna = np.load(paths["dna"], mmap_mode="r")
        self.pos = np.load(paths["pos"], mmap_mode="r")
        chrom_raw = np.load(paths["chrom"], mmap_mode="r", allow_pickle=True)
        self.chrom = np.asarray([canonical_chrom(x) for x in chrom_raw])

        self._validate_shapes(expected_dna_window)
        self.n_cells = int(self.meth.shape[1])
        self.chrom_set = sorted(set(self.chrom.tolist()), key=chrom_sort_key)

        # This channel is intentionally strict: the processed dataset must be a
        # true single-chromosome dataset and must match the explicitly requested
        # chromosome.  We never silently discard additional chromosomes.
        if len(self.chrom_set) != 1:
            raise ValueError(
                "within_chromosome mode requires processed data containing exactly one "
                f"chromosome, found {self.chrom_set}."
            )
        if self.chrom_set[0] != self.split_chrom:
            raise ValueError(
                f"split_chrom={self.split_chrom} but processed data contains chromosome "
                f"{self.chrom_set[0]}."
            )

        self.metadata = self._load_metadata(metadata_file)
        self.pos_base = int(self.metadata.get("pos_base", 0)) if self.metadata else 0
        self.chr_lengths = self._resolve_chromosome_lengths(reference_lengths_file)
        self.pos_norm = self._build_pos_norm()

        chrom_indices = np.flatnonzero(self.chrom == self.split_chrom).astype(np.int64)
        self.split_indices = split_indices_by_position(
            chrom_indices,
            self.pos,
            self.split_fractions,
        )
        self.region_indices = self.split_indices[self.split]
        self.segments = self._build_segments()
        if max_segments is not None:
            self.segments = self.segments[: int(max_segments)]
        if not self.segments:
            raise RuntimeError(f"No segments built for split={self.split}.")

        split_counts = {name: int(len(idx)) for name, idx in self.split_indices.items()}
        core_sites = sum(spec.core_size for spec in self.segments)
        print(
            f"[{self.split}] within_chromosome chr={self.split_chrom}; "
            f"fractions={self.split_fractions}; split_sites={split_counts}; "
            f"segments={len(self.segments)}; core_sites={core_sites}; cells={self.n_cells}; "
            f"strategy={self.segment_strategy}; "
            f"halo={self.context_window if self.segment_strategy == 'overlap' else 0}; "
            f"position_norm={self.position_normalization}"
        )

    def _build_segments(self):
        """Build segments only inside the already-selected split region.

        Because halo is clipped to ``region_indices``, train/validation/test can
        never share overlap context.  The inherited local-neighbor builder then
        operates only inside each such context segment as well.
        """
        specs = []
        region_idx = np.asarray(self.region_indices, dtype=np.int64)
        if len(region_idx) == 0:
            return specs

        region_pos = np.asarray(self.pos[region_idx], dtype=np.float64)
        if len(region_pos) > 1 and np.any(np.diff(region_pos) <= 0):
            raise RuntimeError(
                f"{self.split} region positions are not strictly increasing after splitting."
            )

        halo = self.context_window if self.segment_strategy == "overlap" else 0
        for core_start_region in range(0, len(region_idx), self.segment_size):
            core_end_region = min(core_start_region + self.segment_size, len(region_idx))
            context_start_region = max(0, core_start_region - halo)
            context_end_region = min(len(region_idx), core_end_region + halo)

            context_idx = region_idx[context_start_region:context_end_region].astype(np.int64)
            core_start_local = core_start_region - context_start_region
            core_end_local = core_start_local + (core_end_region - core_start_region)
            specs.append(
                SegmentSpec(
                    chrom=self.split_chrom,
                    context_indices=context_idx,
                    core_start=int(core_start_local),
                    core_end=int(core_end_local),
                )
            )
        return specs
