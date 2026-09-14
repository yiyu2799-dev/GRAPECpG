#!/usr/bin/env python3
import argparse
import random

import numpy as np
import torch

from config import DATA_DEFAULTS, MODEL_DEFAULTS, STAGE_DEFAULTS, TRAIN_DEFAULTS
from data.dataset import canonical_chrom
from data.splitters import validate_split_fractions
from training.trainer import train_model


def _add_bool_pair(parser, name, default, help_true, help_false=None):
    dest = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f'--{name}', dest=dest, action='store_true', help=help_true)
    group.add_argument(
        f'--no-{name}', dest=dest, action='store_false',
        help=help_false or f'Disable {name.replace("-", " ")}.'
    )
    parser.set_defaults(**{dest: bool(default)})


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train GRAPE-CpG for single-cell DNA methylation imputation.'
    )

    parser.add_argument('--dataset_name', default='dataset')
    parser.add_argument('--processed_dir', required=True)
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'stage3', 'end_to_end'], default='stage1')

    # Data / chromosome split.
    parser.add_argument('--meth_file', default=DATA_DEFAULTS['meth_file'])
    parser.add_argument('--dna_file', default=DATA_DEFAULTS['dna_file'])
    parser.add_argument('--pos_file', default=DATA_DEFAULTS['pos_file'])
    parser.add_argument('--chrom_file', default=DATA_DEFAULTS['chrom_file'])
    parser.add_argument('--metadata_file', default=DATA_DEFAULTS['metadata_file'])
    parser.add_argument('--reference_lengths_file', default=None,
                        help='Optional JSON chromosome-length override. Metadata chromosome_lengths is preferred.')
    parser.add_argument(
        '--split_mode', choices=['chromosome_holdout', 'within_chromosome'],
        default=DATA_DEFAULTS['split_mode'],
        help=(
            'Data split protocol. chromosome_holdout preserves the original val/test chromosome logic; '
            'within_chromosome splits one explicitly selected chromosome into contiguous position-ordered regions.'
        ),
    )
    parser.add_argument(
        '--split_chrom', default=DATA_DEFAULTS['split_chrom'],
        help='within_chromosome only: chromosome to split (for example 18 or chr18). Must be provided explicitly.',
    )
    parser.add_argument(
        '--split_fractions', nargs=3, type=float, default=DATA_DEFAULTS['split_fractions'],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help=(
            'within_chromosome only: train/val/test fractions by CpG-site count, for example '
            '--split_fractions 0.8 0.1 0.1. Must be provided explicitly and sum to 1.'
        ),
    )
    parser.add_argument('--val_chrom', default=DATA_DEFAULTS['val_chrom'])
    parser.add_argument('--test_chrom', default=DATA_DEFAULTS['test_chrom'])
    parser.add_argument('--segment_size', type=int, default=DATA_DEFAULTS['segment_size'],
                        help='Number of core CpG sites per segment; adjust by dataset/cell count and GPU memory.')
    parser.add_argument('--segment_strategy', choices=['overlap', 'legacy'],
                        default=DATA_DEFAULTS['segment_strategy'],
                        help='overlap uses halo CpGs around the core segment; legacy reproduces non-overlap segmentation.')
    parser.add_argument('--position_normalization', choices=['reference_length', 'observed_max'],
                        default=DATA_DEFAULTS['position_normalization'])
    parser.add_argument('--max_segments', type=int, default=999999)
    parser.add_argument('--max_val_segments', type=int, default=999999)
    parser.add_argument('--max_test_segments', type=int, default=999999)

    # Core model.
    parser.add_argument('--node_dim', type=int, default=MODEL_DEFAULTS['node_dim'])
    parser.add_argument('--edge_dim', type=int, default=MODEL_DEFAULTS['edge_dim'])
    parser.add_argument('--gnn_layers', type=int, default=MODEL_DEFAULTS['gnn_layers'])
    parser.add_argument('--aggr', default=MODEL_DEFAULTS['aggr'])
    parser.add_argument('--gnn_activation', default=MODEL_DEFAULTS['gnn_activation'])
    parser.add_argument('--post_mlp_hidden', type=int, default=MODEL_DEFAULTS['post_mlp_hidden'])
    parser.add_argument('--dna_window', type=int, default=MODEL_DEFAULTS['dna_window'])
    _add_bool_pair(
        parser, 'dna', MODEL_DEFAULTS['use_dna'],
        'Use the DNA-sequence CNN encoder (default).',
        'Disable DNA-sequence features while retaining Fourier genomic-position features (DNA ablation).'
    )
    parser.add_argument('--fourier_dim', type=int, default=MODEL_DEFAULTS['fourier_dim'])
    parser.add_argument('--cell_emb_dim', type=int, default=MODEL_DEFAULTS['cell_emb_dim'])
    _add_bool_pair(
        parser, 'cell-embedding', MODEL_DEFAULTS['use_cell_embedding'],
        'Use learnable cell identity embeddings (default).',
        'Use GRAPE-style all-one cell features only (cell-embedding ablation).'
    )

    # Learnable local branch. Stage1 disables it automatically; stage2/3/end_to_end enable it unless --no-local.
    _add_bool_pair(parser, 'local', True, 'Enable learnable local CpG context.', 'Disable the local branch.')
    parser.add_argument('--local_windows', default=MODEL_DEFAULTS['local_windows'])
    parser.add_argument(
        '--local_aggregation', choices=['attention', 'mean'],
        default=MODEL_DEFAULTS['local_aggregation'],
        help=(
            'Local aggregation rule: attention=target-conditioned attention (default); '
            'mean=uniform masked mean over the same valid local tokens (attention ablation).'
        ),
    )
    parser.add_argument('--local_meth_dim', type=int, default=MODEL_DEFAULTS['local_meth_dim'])
    parser.add_argument('--local_mask_dim', type=int, default=MODEL_DEFAULTS['local_mask_dim'])
    parser.add_argument('--local_rel_pos_dim', type=int, default=MODEL_DEFAULTS['local_rel_pos_dim'])
    parser.add_argument('--local_hidden_dim', type=int, default=MODEL_DEFAULTS['local_hidden_dim'])
    parser.add_argument('--local_context_dim', type=int, default=MODEL_DEFAULTS['local_context_dim'])
    parser.add_argument('--local_distance_scale_bp', type=float, default=MODEL_DEFAULTS['local_distance_scale_bp'])
    _add_bool_pair(
        parser, 'include-local-center', MODEL_DEFAULTS['include_local_center'],
        'Keep the target CpG itself as an unknown center token in local attention (V11 behavior).',
        'Use surrounding CpGs only; the target remains present in the query/decoder.'
    )

    # Decoder and staged initialization.
    parser.add_argument('--impute_hiddens', default=MODEL_DEFAULTS['impute_hiddens'])
    parser.add_argument('--impute_activation', default='relu')
    parser.add_argument('--decoder_init_strategy', choices=['d1', 'd2', 'd3'], default='d3',
                        help='Stage2 only: d1=V11 shape-compatible warm start; d2=reset decoder; d3=expand first layer and preserve Stage1 global columns.')
    parser.add_argument('--d3_new_weight_init', choices=['random', 'zero'], default='random')
    parser.add_argument(
        '--stage2_global_mode', choices=['frozen', 'small_lr', 'finetune'],
        default=TRAIN_DEFAULTS['stage2_global_mode'],
        help=(
            'Stage2 only: frozen=V11 behavior; small_lr=train the inherited global branch '
            'with lr*global_lr_scale; finetune=train global/local/decoder with the same lr.'
        ),
    )
    parser.add_argument(
        '--global_lr_scale', type=float, default=TRAIN_DEFAULTS['global_lr_scale'],
        help='Stage2 small_lr only: global branch LR = lr * global_lr_scale.'
    )
    parser.add_argument('--load_checkpoint', default=None)

    # Training / optimization. Kept configurable for later tuning.
    parser.add_argument('--epochs', type=int, default=None,
                        help='If omitted, use the selected stage default.')
    parser.add_argument('--lr', type=float, default=None,
                        help='If omitted, use the selected stage default; never hard-coded in the optimizer.')
    parser.add_argument('--opt', choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'],
                        default=TRAIN_DEFAULTS['opt'])
    parser.add_argument('--opt_scheduler', choices=['none', 'step', 'cos'], default=TRAIN_DEFAULTS['opt_scheduler'])
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=TRAIN_DEFAULTS['weight_decay'])
    parser.add_argument('--dropout', type=float, default=MODEL_DEFAULTS['dropout'])
    parser.add_argument('--mask_ratio', type=float, default=TRAIN_DEFAULTS['mask_ratio'])
    _add_bool_pair(
        parser, 'dynamic-train-mask', TRAIN_DEFAULTS['dynamic_train_mask'],
        'Resample training target edges every epoch (default).',
        'Keep the same training target mask across epochs.'
    )
    parser.add_argument('--loss_type', choices=['bce', 'focal'], default=TRAIN_DEFAULTS['loss_type'])
    parser.add_argument('--focal_alpha', type=float, default=TRAIN_DEFAULTS['focal_alpha'])
    parser.add_argument('--focal_gamma', type=float, default=TRAIN_DEFAULTS['focal_gamma'])
    parser.add_argument('--grad_clip', type=float, default=TRAIN_DEFAULTS['grad_clip'])
    parser.add_argument('--seed', type=int, default=TRAIN_DEFAULTS['seed'])
    _add_bool_pair(parser, 'early-stop', TRAIN_DEFAULTS['early_stop'], 'Enable early stopping.', 'Disable early stopping.')
    parser.add_argument('--patience', type=int, default=TRAIN_DEFAULTS['patience'])
    parser.add_argument('--min_delta', type=float, default=TRAIN_DEFAULTS['min_delta'])

    # Runtime / logging.
    parser.add_argument('--log_dir', default='run')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    _add_bool_pair(parser, 'pin-memory', False, 'Use pinned host memory.', 'Disable pinned host memory.')
    parser.add_argument('--print_every', type=int, default=100)
    _add_bool_pair(parser, 'save-model', True, 'Save best validation-AUROC checkpoint.', 'Do not save checkpoints.')
    parser.add_argument('--run_final_test', action='store_true', default=False,
                        help='Evaluate the final best-validation-AUROC checkpoint on the test split once after training.')

    return parser


def _resolve_stage_defaults(args):
    stage_defaults = STAGE_DEFAULTS[args.stage]
    if args.epochs is None:
        args.epochs = int(stage_defaults['epochs'])
    if args.lr is None:
        args.lr = float(stage_defaults['lr'])

    # Stage1 is global-only by definition. Other stages use local by default but
    # can still disable it explicitly for ablation.
    if args.stage == 'stage1':
        args.local = False

    # Final testing is automatic for stage3/end_to_end unless explicitly omitted
    # from a custom workflow by invoking trainer code directly.
    if args.stage in {'stage3', 'end_to_end'}:
        args.run_final_test = True if not args.run_final_test else args.run_final_test



def _resolve_split_config(args):
    """Validate split-specific CLI options without changing legacy split semantics."""
    if args.split_mode == 'chromosome_holdout':
        if args.split_chrom is not None or args.split_fractions is not None:
            raise ValueError(
                '--split_chrom/--split_fractions are only valid with '
                '--split_mode within_chromosome.'
            )
        return

    if args.split_mode != 'within_chromosome':
        raise ValueError(f'Unknown split_mode: {args.split_mode}')
    if args.split_chrom in {None, ''}:
        raise ValueError('--split_chrom is required with --split_mode within_chromosome.')
    if args.split_fractions is None:
        raise ValueError('--split_fractions TRAIN VAL TEST is required with --split_mode within_chromosome.')

    args.split_chrom = canonical_chrom(args.split_chrom)
    args.split_fractions = list(validate_split_fractions(args.split_fractions))

def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = build_parser()
    args = parser.parse_args()
    _resolve_stage_defaults(args)
    _resolve_split_config(args)
    _seed_everything(args.seed)
    train_model(args)


if __name__ == '__main__':
    main()
