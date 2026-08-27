import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def focal_loss_with_logits(logits, labels, alpha=0.75, gamma=2.0, reduction='mean'):
    labels = labels.float()
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    prob = torch.sigmoid(logits)
    p_t = prob * labels + (1.0 - prob) * (1.0 - labels)
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    loss = alpha_t * ((1.0 - p_t) ** gamma) * bce
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    return loss.mean()


def loss_for_logits(logits, labels, loss_type='bce', focal_alpha=0.75, focal_gamma=2.0, reduction='mean'):
    if loss_type == 'focal':
        return focal_loss_with_logits(
            logits, labels, alpha=focal_alpha, gamma=focal_gamma, reduction=reduction
        )
    return F.binary_cross_entropy_with_logits(logits, labels.float(), reduction=reduction)


def binary_metrics_from_logits(logits, labels):
    with torch.no_grad():
        labels = labels.float()
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        acc = (pred == labels).float().mean().item()

        tp = int(((pred == 1) & (labels == 1)).sum().item())
        tn = int(((pred == 0) & (labels == 0)).sum().item())
        fp = int(((pred == 1) & (labels == 0)).sum().item())
        fn = int(((pred == 0) & (labels == 1)).sum().item())
        precision = tp / float(tp + fp + 1e-8)
        recall = tp / float(tp + fn + 1e-8)
        specificity = tn / float(tn + fp + 1e-8)
        f1 = 2.0 * precision * recall / float(precision + recall + 1e-8)
        denom = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        y_true = labels.detach().cpu().numpy()
        y_score = prob.detach().cpu().numpy()
        if len(set(y_true.tolist())) < 2:
            auroc = float('nan')
            auprc = float('nan')
        else:
            auroc = float(roc_auc_score(y_true, y_score))
            auprc = float(average_precision_score(y_true, y_score))

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'mcc': mcc,
        'auroc': auroc,
        'auprc': auprc,
        'pos_rate': labels.mean().item(),
        'pred_pos_rate': pred.mean().item(),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def finite_metric(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False
