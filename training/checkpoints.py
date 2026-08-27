import os.path as osp

import torch


GLOBAL_PREFIXES = (
    'dna_encoder.',
    'site_fusion.',
    'cell_embedding.',
    'cell_fusion.',
    'edge_encoder.',
    'gnn.',
)
DECODER_PREFIX = 'edge_decoder.'
FIRST_DECODER_WEIGHT = 'edge_decoder.layers.0.0.weight'


STRICT_CONFIG_KEYS = (
    'node_dim', 'edge_dim', 'gnn_layers', 'aggr', 'gnn_activation',
    'post_mlp_hidden', 'dna_window', 'fourier_dim', 'cell_emb_dim',
    'cell_embedding', 'local', 'local_windows', 'local_meth_dim',
    'local_mask_dim', 'local_rel_pos_dim', 'local_hidden_dim',
    'local_context_dim', 'local_distance_scale_bp', 'include_local_center',
    'impute_hiddens', 'impute_activation', 'segment_size', 'segment_strategy',
    'position_normalization',
)

STAGE2_GLOBAL_CONFIG_KEYS = (
    'node_dim', 'edge_dim', 'gnn_layers', 'aggr', 'gnn_activation',
    'post_mlp_hidden', 'dna_window', 'fourier_dim', 'cell_emb_dim',
    'cell_embedding', 'local_windows', 'segment_size', 'segment_strategy',
    'position_normalization',
)


def _validate_embedded_config(raw, expected_config, keys=STRICT_CONFIG_KEYS, context='checkpoint'):
    """Reject silent semantic mismatches when a checkpoint embeds its run config."""
    if expected_config is None or not isinstance(raw, dict) or not isinstance(raw.get('config'), dict):
        return
    source = raw['config']
    mismatches = []
    for key in keys:
        if key not in source or key not in expected_config:
            continue
        if source[key] != expected_config[key]:
            mismatches.append(f'{key}: checkpoint={source[key]!r}, current={expected_config[key]!r}')
    if mismatches:
        raise RuntimeError(f'{context} configuration mismatch: ' + '; '.join(mismatches[:20]))


def _unwrap_checkpoint(raw):
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        state = raw['model_state_dict']
    elif isinstance(raw, dict) and 'state_dict' in raw:
        state = raw['state_dict']
    else:
        state = raw
    if not isinstance(state, dict):
        raise TypeError('Checkpoint does not contain a PyTorch state_dict.')
    if state and all(str(k).startswith('module.') for k in state):
        state = {str(k)[7:]: value for k, value in state.items()}
    return state


def load_raw_checkpoint(path, device):
    if not path or not osp.exists(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    raw = torch.load(path, map_location=device)
    return raw, _unwrap_checkpoint(raw)


def global_keys(model):
    return {key for key in model.state_dict() if key.startswith(GLOBAL_PREFIXES)}


def _require_global_compatible(model, source_state):
    current = model.state_dict()
    required = global_keys(model)
    errors = []
    for key in sorted(required):
        if key not in source_state:
            errors.append(f'missing {key}')
        elif tuple(source_state[key].shape) != tuple(current[key].shape):
            errors.append(
                f'{key}: checkpoint{tuple(source_state[key].shape)} != model{tuple(current[key].shape)}'
            )
    if errors:
        raise RuntimeError('Global checkpoint incompatibility: ' + '; '.join(errors[:20]))
    return required


def _copy_matching(current, source_state, predicate=lambda _key: True):
    copied = []
    for key, value in source_state.items():
        if not predicate(key) or key not in current:
            continue
        if tuple(current[key].shape) == tuple(value.shape):
            current[key] = value
            copied.append(key)
    return copied


def load_stage2(model, checkpoint_path, device, strategy='d3', d3_new_weight_init='random', expected_config=None):
    """Load Stage1 -> Stage2 with an explicit decoder initialization strategy.

    d1: reproduce V11 shape-compatible warm start.
    d2: load global branch only; reset the entire Stage2 decoder.
    d3: load global branch, preserve the old first-layer global columns, initialize
        only the newly added local columns, and warm-start later compatible decoder layers.
    """
    raw, source = load_raw_checkpoint(checkpoint_path, device)
    _validate_embedded_config(
        raw, expected_config, keys=STAGE2_GLOBAL_CONFIG_KEYS, context='Stage1 -> Stage2'
    )
    current = model.state_dict()
    required_global = _require_global_compatible(model, source)

    copied_global = _copy_matching(current, source, lambda key: key in required_global)
    copied_decoder = []

    if strategy == 'd1':
        copied_decoder = _copy_matching(current, source, lambda key: key.startswith(DECODER_PREFIX))

    elif strategy == 'd2':
        pass

    elif strategy == 'd3':
        # Warm-start all decoder tensors whose shapes are unchanged.
        copied_decoder = _copy_matching(
            current,
            source,
            lambda key: key.startswith(DECODER_PREFIX) and key != FIRST_DECODER_WEIGHT,
        )

        if FIRST_DECODER_WEIGHT not in source or FIRST_DECODER_WEIGHT not in current:
            raise RuntimeError('D3 requires a decoder first linear layer in both checkpoints.')
        old_weight = source[FIRST_DECODER_WEIGHT]
        new_weight = current[FIRST_DECODER_WEIGHT].clone()
        if old_weight.ndim != 2 or new_weight.ndim != 2:
            raise RuntimeError('D3 decoder first-layer weights must be matrices.')
        if old_weight.shape[0] != new_weight.shape[0] or old_weight.shape[1] >= new_weight.shape[1]:
            raise RuntimeError(
                'D3 expects Stage2 first-layer input width to expand: '
                f'old={tuple(old_weight.shape)}, new={tuple(new_weight.shape)}.'
            )
        if d3_new_weight_init == 'zero':
            new_weight.zero_()
        elif d3_new_weight_init != 'random':
            raise ValueError('d3_new_weight_init must be random or zero.')
        new_weight[:, : old_weight.shape[1]] = old_weight
        current[FIRST_DECODER_WEIGHT] = new_weight
        copied_decoder.append(FIRST_DECODER_WEIGHT + f' [expanded {old_weight.shape[1]}->{new_weight.shape[1]}]')
    else:
        raise ValueError('decoder_init_strategy must be d1, d2, or d3.')

    model.load_state_dict(current, strict=True)
    print(f'Stage2 load strategy={strategy}; global tensors={len(copied_global)}; decoder transfers={len(copied_decoder)}')
    for item in copied_decoder[:20]:
        print('  decoder:', item)


def load_strict(model, checkpoint_path, device, expected_config=None):
    raw, source = load_raw_checkpoint(checkpoint_path, device)
    _validate_embedded_config(raw, expected_config)
    current = model.state_dict()
    missing = sorted(set(current) - set(source))
    unexpected = sorted(set(source) - set(current))
    mismatch = sorted(
        key for key in set(current) & set(source)
        if tuple(current[key].shape) != tuple(source[key].shape)
    )
    if missing or unexpected or mismatch:
        raise RuntimeError(
            f'Strict checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}, '
            f'shape_mismatch={mismatch[:10]}'
        )
    model.load_state_dict(source, strict=True)
    print(f'Strict checkpoint load passed: {len(source)} tensors.')


def save_checkpoint(path, model, args, epoch, val_metrics, optimizer=None, scheduler=None):
    payload = {
        'format_version': 2,
        'model_state_dict': model.state_dict(),
        'config': dict(vars(args)),
        'epoch': int(epoch),
        'best_val_auroc': float(val_metrics['auroc']),
        'val_metrics': dict(val_metrics),
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(payload, path)
