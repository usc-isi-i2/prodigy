"""Single-episode support-only calibration of the nominated value intervention."""
import numpy as np
import torch
from .class_reference_kv import kv_forward
from .replay import episode_probe
from .support_boundary_folds import balanced_support_folds, hide_support_labels
from .support_boundary_calibration import fit_calibrator, apply_calibrator


def outcomes(model, batch):
    native = kv_forward(model, batch)
    donor = kv_forward(model, batch, alpha=0.)
    value = kv_forward(model, batch, value_donor=donor[4]['kqv_native'])
    q = native[4]['query_mask']
    torch.testing.assert_close(native[4]['final_inputs'][q], value[4]['final_inputs'][q], rtol=0, atol=0)
    return {'native': native[0], 'value': value[0],
            'u1_ridge': episode_probe(native[1]['U1_pre_meta'], batch, 'ridge')}


def evaluate_episode(model, batch):
    if batch[0].task_id_per_sample.unique().numel() != 1:
        raise ValueError('exactly one episode required')
    original_q = batch[5].reshape(-1, 2)[:, 0].bool()
    support = torch.where(~original_q)[0]
    truth = batch[2][support].argmax(1).numpy()
    # Hide original query truth even in the uncalibrated full forward.
    clean = list(batch)
    clean[2] = batch[2].clone()
    clean[2][original_q] = 0
    clean[2][original_q, 0] = 1
    full = outcomes(model, tuple(clean))
    oof = {name: np.full(len(batch[2]), np.nan) for name in full}
    for heldout in balanced_support_folds(batch):
        folded = hide_support_labels(tuple(clean), heldout)
        predictions = outcomes(model, folded)
        query_rows = torch.where(folded[5].reshape(-1, 2)[:, 0].bool())[0]
        positions = torch.searchsorted(query_rows, heldout)
        for name, logits in predictions.items():
            oof[name][heldout.numpy()] = (logits[positions, 1] - logits[positions, 0]).double().numpy()
    result, fits = dict(full), {}
    for name, logits in full.items():
        margins = oof[name][support.numpy()]
        if not np.isfinite(margins).all():
            raise ValueError('incomplete out-of-fold support predictions')
        fit = fit_calibrator(margins, truth)
        calibrated = apply_calibrator((logits[:, 1] - logits[:, 0]).double().numpy(), fit)
        # The calibrator returns log-odds, not probabilities.
        calibrated = torch.as_tensor(calibrated, dtype=torch.float64)
        result[name + '_calibrated'] = torch.stack((torch.zeros_like(calibrated), calibrated), 1)
        fits[name] = fit
    return {'logits': result, 'calibration': fits,
            'support_oof_margins': {k: v[support.numpy()] for k, v in oof.items()},
            'support_labels': truth, 'query_labels_used_for_fit': False}
