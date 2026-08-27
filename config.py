"""Central defaults for the GRAPE-CpG paper/reference implementation.

Architecture-defining defaults match the validated model. Experiment-sensitive
parameters remain configurable from the command line.
"""

MODEL_DEFAULTS = {
    "node_dim": 64,
    "edge_dim": 64,
    "gnn_layers": 3,
    "aggr": "mean",
    "gnn_activation": "relu",
    "post_mlp_hidden": 64,
    "dna_window": 201,
    "fourier_dim": 16,
    "cell_emb_dim": 8,
    "use_cell_embedding": True,
    "local_windows": "5_20",
    "local_meth_dim": 8,
    "local_mask_dim": 4,
    "local_rel_pos_dim": 8,
    "local_hidden_dim": 128,
    "local_context_dim": 32,
    "local_distance_scale_bp": 1000.0,
    "include_local_center": True,
    "impute_hiddens": "128_64",
    "dropout": 0.1,
}

TRAIN_DEFAULTS = {
    "mask_ratio": 0.15,
    "dynamic_train_mask": True,
    "opt": "adam",
    "opt_scheduler": "none",
    "weight_decay": 1e-4,
    "lr": 5e-4,
    "grad_clip": 1.0,
    "seed": 1,
    "epochs": 50,
    "early_stop": True,
    "patience": 10,
    "min_delta": 1e-4,
    "loss_type": "bce",
    "stage2_global_mode": "frozen",
    "global_lr_scale": 0.1,
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
}

DATA_DEFAULTS = {
    "meth_file": "meth_matrix.npy",
    "dna_file": "dna_windows_centerC.npy",
    "pos_file": "pos.npy",
    "chrom_file": "chrom.npy",
    "metadata_file": "metadata.json",
    "val_chrom": "5",
    "test_chrom": "10",
    "segment_size": 1024,
    "segment_strategy": "overlap",
    "position_normalization": "reference_length",
}

STAGE_DEFAULTS = {
    "stage1": {"epochs": 80, "lr": 5e-4},
    "stage2": {"epochs": 60, "lr": 7e-4},
    "stage3": {"epochs": 50, "lr": 1e-4},
    "end_to_end": {"epochs": 80, "lr": 5e-4},
}


def parse_int_tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    text = str(value).strip()
    if not text:
        return tuple()
    return tuple(int(v) for v in text.split("_") if v != "")
