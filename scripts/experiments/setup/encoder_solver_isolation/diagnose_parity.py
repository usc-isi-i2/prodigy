"""One real training batch: CUDA null-replay versus gradient-path parity.

Use the ordinary run_single_experiment arguments plus --diagnostic-output PATH.
This loads the graph/trainer once and performs three independent one-step probes;
it does not train or alter a checkpoint. No deterministic settings are added.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

from experiments.run_single_experiment import get_params, seed_everything, load_dataset
from experiments.trainer import TrainerFS


def hashes(value, prefix="batch"):
    """Hash every tensor input, including feature values and supernode edges."""
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous()
        return {prefix: dict(shape=list(array.shape), dtype=str(array.dtype),
                            sha256=hashlib.sha256(array.numpy().tobytes()).hexdigest())}
    if hasattr(value, "to_dict"):
        return hashes(value.to_dict(), prefix)
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            result.update(hashes(value[key], f"{prefix}.{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(hashes(item, f"{prefix}.{index}"))
        return result
    return {prefix: dict(value=repr(value))}


def compare(a, b):
    result = {}
    for key in a:
        x, y = a[key], b[key]
        if x is None or y is None:
            result[key] = dict(exact=x is None and y is None,
                               none_a=x is None, none_b=y is None)
        else:
            delta = x.double() - y.double()
            result[key] = dict(exact=torch.equal(x, y),
                               max_abs=float(delta.abs().max()) if delta.numel() else 0.,
                               l2=float(delta.norm()), norm_a=float(x.double().norm()),
                               norm_b=float(y.double().norm()))
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--diagnostic-output', type=Path, required=True)
    own, remaining = parser.parse_known_args()
    if own.diagnostic_output.exists():
        raise ValueError('new diagnostic output required')
    sys.argv = [sys.argv[0], *remaining]
    torch.set_num_threads(4)
    params = get_params()
    seed_everything(params)
    trainer = TrainerFS(load_dataset(params), params)
    if params['layers'] != 'S,U,M' or params['task_name'] != 'neighbor_matching':
        raise ValueError('diagnostic requires the isolation NM architecture')
    batch = next(iter(trainer.train_dataloader))
    model = trainer.model
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    records = {}
    encoder = lambda key: key.startswith(('layer_list.0.', 'layer_list.1.', 'initial_input_mlp.'))
    for name, mode in [('isolated_a', 'isolated'), ('isolated_b', 'isolated'), ('ridge_only', 'ridge_only')]:
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        model.params['encoder_solver_objective'] = mode
        model.train()
        model.encoder_solver_training = True
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        inputs = [item.to(trainer.device) for item in copy.deepcopy(batch)]
        before = hashes(inputs)
        initial_exact = all(torch.equal(v.detach().cpu(), initial[k]) for k, v in model.state_dict().items())
        pooled = []
        pool = model.layer_list[1]
        pool_forward = pool.forward
        def capture_pool(*args, **kwargs):
            output = pool_forward(*args, **kwargs)
            pooled.append(output.detach().cpu().clone())
            return output
        pool.forward = capture_pool
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=trainer.learning_rate, weight_decay=params['weight_decay'])
        try:
            yt, yp, _ = model(*inputs)
            native, _ = trainer.get_loss_and_acc(yt, yp)
            ridge = model.encoder_solver_ridge_loss
            (ridge if mode == 'ridge_only' else native + ridge).backward()
        finally:
            pool.forward = pool_forward
            model.encoder_solver_training = False
        gradients = {k: None if p.grad is None else p.grad.detach().cpu().clone()
                     for k, p in model.named_parameters() if encoder(k)}
        optimizer.step()
        records[name] = dict(inputs=before, initial_exact=initial_exact,
                             native_loss=float(native.detach()), ridge_loss=float(ridge.detach()),
                             gradients=gradients, outputs={'u1': pooled[0], 'full_logits': yp.detach().cpu()},
                             state={k: v.detach().cpu().clone() for k, v in model.state_dict().items() if encoder(k)})
    result = dict(device=str(trainer.device), initial_states_exact=all(r['initial_exact'] for r in records.values()),
                  input_hashes=records['isolated_a']['inputs'],
                  inputs_exact=all(r['inputs'] == records['isolated_a']['inputs'] for r in records.values()),
                  losses={k: {m: r[m] for m in ('native_loss', 'ridge_loss')} for k, r in records.items()}, comparisons={})
    for name in ('isolated_b', 'ridge_only'):
        result['comparisons'][name] = {field: compare(records['isolated_a'][field], records[name][field])
                                      for field in ('gradients', 'state', 'outputs')}
    own.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
    own.diagnostic_output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(own.diagnostic_output),
                          initial_states_exact=result['initial_states_exact'],
                          inputs_exact=result['inputs_exact'], losses=result['losses'],
                          comparisons={name: {field: dict(
                              exact=all(row['exact'] for row in tensors.values()),
                              max_abs=max((row.get('max_abs', 0.) for row in tensors.values()), default=0.))
                              for field, tensors in fields.items()}
                              for name, fields in result['comparisons'].items()}), indent=2))


if __name__ == '__main__':
    main()
