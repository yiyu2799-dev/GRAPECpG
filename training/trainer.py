import json
import os
import os.path as osp
import pickle
import platform
import sys

import torch
import torch.optim as optim
import torch_geometric
from torch_geometric.loader import DataLoader

from config import parse_int_tuple
from data.dataset import MethylationGenomeDataset
from models.model import GrapeCpGModel
from training.checkpoints import load_stage2, load_strict, save_checkpoint
from training.metrics import binary_metrics_from_logits, finite_metric, loss_for_logits


def _optimizer_class_and_kwargs(args):
    if args.opt == 'adam':
        return optim.Adam, {}
    if args.opt == 'adamw':
        return optim.AdamW, {'betas': (0.95, 0.9)}
    if args.opt == 'sgd':
        return optim.SGD, {'momentum': 0.95}
    if args.opt == 'rmsprop':
        return optim.RMSprop, {}
    if args.opt == 'adagrad':
        return optim.Adagrad, {}
    raise ValueError(f'Unsupported optimizer: {args.opt}')


def _make_param_groups(args, model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError('No trainable parameters.')

    if args.stage == 'stage2' and args.stage2_global_mode == 'small_lr':
        if args.global_lr_scale <= 0:
            raise ValueError('global_lr_scale must be > 0 for stage2_global_mode=small_lr.')
        global_params = [p for p in model.global_parameters() if p.requires_grad]
        global_ids = {id(p) for p in global_params}
        other_params = [p for p in trainable if id(p) not in global_ids]
        if not global_params or not other_params:
            raise RuntimeError('small_lr Stage2 requires both global and local/decoder trainable parameters.')
        return [
            {'params': global_params, 'lr': args.lr * args.global_lr_scale, 'group_name': 'global'},
            {'params': other_params, 'lr': args.lr, 'group_name': 'local_decoder'},
        ]

    return [{'params': trainable, 'lr': args.lr, 'group_name': 'all_trainable'}]


def build_optimizer(args, model):
    optimizer_cls, extra_kwargs = _optimizer_class_and_kwargs(args)
    param_groups = _make_param_groups(args, model)
    optimizer = optimizer_cls(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        **extra_kwargs,
    )

    if args.opt_scheduler == 'none':
        scheduler = None
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate
        )
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise ValueError(f'Unsupported scheduler: {args.opt_scheduler}')
    return optimizer, scheduler


def build_dataset(args, split, seed, max_segments):
    context_window = max(parse_int_tuple(args.local_windows))
    return MethylationGenomeDataset(
        processed_dir=args.processed_dir,
        split=split,
        meth_file=args.meth_file,
        dna_file=args.dna_file,
        pos_file=args.pos_file,
        chrom_file=args.chrom_file,
        metadata_file=args.metadata_file,
        reference_lengths_file=args.reference_lengths_file,
        val_chrom=args.val_chrom,
        test_chrom=args.test_chrom,
        segment_size=args.segment_size,
        segment_strategy=args.segment_strategy,
        context_window=context_window,
        position_normalization=args.position_normalization,
        mask_ratio=args.mask_ratio,
        dynamic_train_mask=args.dynamic_train_mask,
        seed=seed,
        max_segments=max_segments,
        expected_dna_window=args.dna_window,
        use_local=args.local,
        include_local_center=args.include_local_center,
        local_distance_scale_bp=args.local_distance_scale_bp,
    )


def build_loader_kwargs(args):
    if args.num_workers < 0:
        raise ValueError('num_workers must be >= 0.')
    kwargs = {
        'batch_size': 1,
        'num_workers': int(args.num_workers),
        'pin_memory': bool(args.pin_memory),
    }
    if args.num_workers > 0:
        if args.prefetch_factor <= 0:
            raise ValueError('prefetch_factor must be positive.')
        kwargs['prefetch_factor'] = int(args.prefetch_factor)
        # Dynamic masks are updated through dataset.set_epoch; recreate workers
        # each epoch so worker copies see the new epoch value.
        kwargs['persistent_workers'] = False
    return kwargs


def _move(data, device, non_blocking=False):
    return data.to(device, non_blocking=bool(non_blocking))


def evaluate_model(model, loader, device, args, non_blocking=False):
    model.eval()
    total_loss = 0.0
    total_edges = 0
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for data in loader:
            data = _move(data, device, non_blocking)
            logits = model(data)
            labels = data.target_labels.float()
            loss = loss_for_logits(
                logits,
                labels,
                loss_type=args.loss_type,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
                reduction='sum',
            )
            total_loss += float(loss.item())
            total_edges += int(labels.numel())
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())

    if total_edges == 0:
        raise RuntimeError('Evaluation loader produced no target edges.')
    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    metrics = binary_metrics_from_logits(logits_all, labels_all)
    metrics['loss'] = total_loss / total_edges
    return metrics


def _write_metadata(args, log_path):
    with open(osp.join(log_path, 'args.json'), 'w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    with open(osp.join(log_path, 'cmd_input.txt'), 'w', encoding='utf-8') as handle:
        handle.write('python ' + ' '.join(sys.argv) + '\n')

    runtime = {
        'python': platform.python_version(),
        'pytorch': torch.__version__,
        'torch_geometric': getattr(torch_geometric, '__version__', 'unknown'),
        'cuda_runtime': torch.version.cuda,
        'cuda_available': bool(torch.cuda.is_available()),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(osp.join(log_path, 'environment.json'), 'w', encoding='utf-8') as handle:
        json.dump(runtime, handle, indent=2, sort_keys=True)


def _count_trainable(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _format_metrics(prefix, metrics):
    return (
        f'{prefix}_loss={metrics["loss"]:.5f} '
        f'{prefix}_acc={metrics["acc"]:.4f} {prefix}_f1={metrics["f1"]:.4f} '
        f'{prefix}_mcc={metrics["mcc"]:.4f} {prefix}_auroc={metrics["auroc"]:.4f} '
        f'{prefix}_auprc={metrics["auprc"]:.4f}'
    )


def _configure_staged_loading(model, args, device):
    if args.stage == 'stage1':
        if args.load_checkpoint:
            raise ValueError('stage1 does not use --load_checkpoint. Use end_to_end or a staged workflow.')
        return

    if args.stage == 'stage2':
        if not args.load_checkpoint:
            raise ValueError('stage2 requires --load_checkpoint from Stage1.')
        load_stage2(
            model,
            args.load_checkpoint,
            device,
            strategy=args.decoder_init_strategy,
            d3_new_weight_init=args.d3_new_weight_init,
            expected_config=vars(args),
        )
        if args.stage2_global_mode == 'frozen':
            model.freeze_global_branch()
            print('Stage2 global mode=frozen: inherited global branch is frozen; local+decoder are trainable.')
        elif args.stage2_global_mode == 'small_lr':
            if args.global_lr_scale <= 0:
                raise ValueError('global_lr_scale must be > 0 for stage2_global_mode=small_lr.')
            model.unfreeze_global_branch()
            print(
                'Stage2 global mode=small_lr: global/local/decoder are trainable; '
                f'global lr scale={args.global_lr_scale:g}.'
            )
        elif args.stage2_global_mode == 'finetune':
            model.unfreeze_global_branch()
            print('Stage2 global mode=finetune: global/local/decoder are trainable with the same base lr.')
        else:
            raise ValueError(f'Unknown stage2_global_mode: {args.stage2_global_mode}')
        return

    if args.stage == 'stage3':
        if not args.load_checkpoint:
            raise ValueError('stage3 requires --load_checkpoint from Stage2.')
        load_strict(model, args.load_checkpoint, device, expected_config=vars(args))
        print('Stage3: Stage2 checkpoint loaded strictly; all branches are trainable.')
        return

    if args.stage == 'end_to_end':
        if args.load_checkpoint:
            raise ValueError('end_to_end is defined here as training global+local+decoder from scratch.')
        return

    raise ValueError(f'Unknown stage: {args.stage}')


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Resolved stage:', args.stage, 'epochs=', args.epochs, 'lr=', args.lr)

    log_path = osp.join('runs', str(args.dataset_name), str(args.log_dir))
    os.makedirs(log_path, exist_ok=True)
    train_set = build_dataset(args, 'train', args.seed, args.max_segments)
    val_set = build_dataset(args, 'val', args.seed + 100000, args.max_val_segments)
    if train_set.n_cells != val_set.n_cells:
        raise ValueError('Cell count differs between train and validation splits.')
    args.n_cells = int(train_set.n_cells)
    _write_metadata(args, log_path)

    loader_kwargs = build_loader_kwargs(args)
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    non_blocking = bool(args.pin_memory and device.type == 'cuda')

    model = GrapeCpGModel(args).to(device)
    _configure_staged_loading(model, args, device)
    print('Trainable parameters:', _count_trainable(model))

    optimizer, scheduler = build_optimizer(args, model)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    for group in optimizer.param_groups:
        print(
            'Optimizer group:', group.get('group_name', 'unnamed'),
            'lr=', group['lr'], 'params=', sum(p.numel() for p in group['params'])
        )

    history = {
        'train_loss': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_mcc': [],
        'val_precision': [], 'val_recall': [], 'val_specificity': [],
        'val_auroc': [], 'val_auprc': [], 'val_pos_rate': [], 'val_pred_pos_rate': [],
    }

    best_val_auroc = -float('inf')
    best_auroc_epoch = -1
    best_val_metrics = None
    best_state_cpu = None
    best_val_loss = float('inf')
    best_loss_epoch = -1
    no_loss_improve = 0
    checkpoint_path = osp.join(log_path, 'model_best_val_auroc.pt')

    for epoch in range(int(args.epochs)):
        train_set.set_epoch(epoch)
        model.train()
        if args.stage == 'stage2' and args.stage2_global_mode == 'frozen':
            model.set_global_branch_eval()

        total_loss = 0.0
        total_edges = 0
        for step, data in enumerate(train_loader):
            data = _move(data, device, non_blocking)
            logits = model(data)
            labels = data.target_labels.float()
            loss = loss_for_logits(
                logits,
                labels,
                loss_type=args.loss_type,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
                reduction='mean',
            )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
            optimizer.step()

            n_edges = int(labels.numel())
            total_loss += float(loss.item()) * n_edges
            total_edges += n_edges
            if args.print_every > 0 and (step + 1) % args.print_every == 0:
                print(
                    f'epoch={epoch} step={step + 1}/{len(train_loader)} '
                    f'batch_loss={loss.item():.5f} target_edges={n_edges}'
                )

        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / max(total_edges, 1)
        val_metrics = evaluate_model(model, val_loader, device, args, non_blocking)
        history['train_loss'].append(train_loss)
        for key in [
            'loss', 'acc', 'f1', 'mcc', 'precision', 'recall', 'specificity',
            'auroc', 'auprc', 'pos_rate', 'pred_pos_rate',
        ]:
            history[f'val_{key}'].append(val_metrics[key])

        print(f'epoch={epoch} train_loss={train_loss:.5f} ' + _format_metrics('val', val_metrics))

        # Checkpoint selection: validation AUROC only.
        if finite_metric(val_metrics['auroc']) and float(val_metrics['auroc']) > best_val_auroc:
            best_val_auroc = float(val_metrics['auroc'])
            best_auroc_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_state_cpu = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if args.save_model:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    args,
                    epoch,
                    val_metrics,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                print('Saved new best validation-AUROC checkpoint:', checkpoint_path)

        # Early stopping: validation loss only, preserving the V11/MambaCpG-style rule.
        if float(val_metrics['loss']) < best_val_loss - float(args.min_delta):
            best_val_loss = float(val_metrics['loss'])
            best_loss_epoch = epoch
            no_loss_improve = 0
        else:
            no_loss_improve += 1

        if args.early_stop and no_loss_improve >= int(args.patience):
            print(
                f'Early stopping at epoch={epoch}; best val loss={best_val_loss:.5f} '
                f'at epoch={best_loss_epoch}.'
            )
            break

    if best_state_cpu is None:
        raise RuntimeError('No finite validation AUROC was observed; no best model can be selected.')

    final_test_metrics = None
    if args.run_final_test:
        # Use the exact model selected by validation AUROC, never the last epoch.
        model.load_state_dict(best_state_cpu, strict=True)
        test_set = build_dataset(args, 'test', args.seed + 200000, args.max_test_segments)
        if test_set.n_cells != args.n_cells:
            raise ValueError('Cell count differs between train and test splits.')
        test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
        final_test_metrics = evaluate_model(model, test_loader, device, args, non_blocking)
        print('FINAL TEST from best validation-AUROC checkpoint: ' + _format_metrics('test', final_test_metrics))
        with open(osp.join(log_path, 'final_test_metrics.json'), 'w', encoding='utf-8') as handle:
            json.dump(final_test_metrics, handle, indent=2, sort_keys=True)

    result = {
        'args': vars(args),
        'history': history,
        'best_val_auroc': {
            'value': best_val_auroc,
            'epoch': best_auroc_epoch,
            'val_metrics': best_val_metrics,
            'checkpoint': 'model_best_val_auroc.pt' if args.save_model else None,
        },
        'early_stopping_reference': {
            'best_val_loss': best_val_loss,
            'epoch': best_loss_epoch,
        },
        'final_test_metrics': final_test_metrics,
    }
    with open(osp.join(log_path, 'result.pkl'), 'wb') as handle:
        pickle.dump(result, handle)
    with open(osp.join(log_path, 'result_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'best_val_auroc': result['best_val_auroc'],
                'early_stopping_reference': result['early_stopping_reference'],
                'final_test_metrics': final_test_metrics,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print('Training finished.')
    print('Best validation AUROC:', best_val_auroc, 'epoch=', best_auroc_epoch)
    return result
