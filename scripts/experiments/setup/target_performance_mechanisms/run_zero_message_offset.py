"""Fixed support-side MLP(0) subtraction, paired across message contexts.

No fitted multiplier, checkpoint search, or training. Private outputs must not
be committed to the public source repository.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import time

import torch

from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import evaluate_logits
from .contrast_metrics import ranking_metrics
from .message_scale import forward_scaled
from .replay import batch_hash
from .role_context import query_mask
from .run_class_reference_kv import load_cell
from .verify_member_training import model_digest


@contextmanager
def subtract_default(model, batch):
    """Subtract only the parameter-derived message-MLP default before BN."""
    if any(m.training for m in model.modules()):
        raise ValueError('frozen evaluation mode required')
    bg = model.layer_list[0]
    if len(bg.module_list) != 1:
        raise ValueError('one background layer required')
    layer = bg.module_list[0]
    if layer.lin_edge_attr is not None:
        raise ValueError('edge attributes unsupported')
    node_support = ~query_mask(batch)[batch[0].batch]
    with torch.no_grad():
        offset = layer.mlp(layer.lin_x.weight.new_zeros(1, layer.lin_x.out_features)).detach()
    audit = {'calls': 0, 'offset_norm': float(offset.norm()), 'query_input_exact': True}

    def modify(module, args):
        old = args[0]
        new = torch.where(node_support[:, None], old - offset, old)
        torch.testing.assert_close(new[~node_support], old[~node_support], rtol=0, atol=0)
        torch.testing.assert_close(new[node_support], (old-offset)[node_support], rtol=0, atol=0)
        audit['calls'] += 1
        return (new, *args[1:])

    handle = layer.bn.register_forward_pre_hook(modify)
    try:
        yield audit
        if audit['calls'] != 1:
            raise ValueError('unexpected pre-BN call count')
    finally:
        handle.remove()


def dispersion(x):
    u = torch.nn.functional.normalize(x.double(), dim=1)
    return float((u-u.mean(0)).square().sum(1).mean())


def measured_forward(model, batch, alpha):
    saved = {}
    handle = model.final_label_mlp.register_forward_hook(
        lambda m,a,v: saved.update(final_label=v.detach().clone()))
    try:
        logits, trace = forward_scaled(model,batch,alpha=alpha)
        return logits, {**trace, **saved}
    finally:
        handle.remove()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--long-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.phase = 'long_test'
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('fresh output directory and hidden GPUs required')
    if not (args.long_root/'DONE.json').exists():
        raise ValueError('completed parent required')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    conditions = {'intact': (1., False), 'suppressed': (0., False),
                  'intact_minus_default': (1., True), 'suppressed_minus_default': (0., True)}
    protocol = {'revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'parent': str(args.long_root), 'streams': ['original','fresh'],
        'conditions': conditions, 'prediction': 'Subtraction increases suppressed pre-BN support angular dispersion in both streams.',
        'performance_prediction': None, 'selection': 'Existing nominated checkpoint and retained streams; no new search.',
        'intervention': 'Subtract MLP(0), not self bias, from support pre-BN input in both contexts.',
        'interpretation': 'Only suppressed recovery explains an ablation side effect; intact improvements would implicate ordinary inference. Geometry recovery alone is not performance repair.'}
    write_json(args.output/'protocol.json', protocol)
    started = time.monotonic(); metrics=[]; geometry=[]; receipts=[]
    try:
        with torch.no_grad():
            for stream in protocol['streams']:
                model, labels, paths, expected, refs, rec = load_cell(args, stream)
                model.eval(); values={c:[] for c in conditions}
                for bi,path in enumerate(paths):
                    batch=torch.load(path,map_location='cpu',weights_only=False)
                    if batch_hash(batch)!=expected[bi]: raise ValueError('input hash changed')
                    q=query_mask(batch); ids=batch[0].task_id_per_sample
                    centers=batch[0].ptr[:-1]; baseline_query=None
                    for name,(alpha,subtract) in conditions.items():
                        if subtract:
                            with subtract_default(model,batch) as audit:
                                logits,trace=measured_forward(model,batch,alpha)
                        else:
                            logits,trace=measured_forward(model,batch,alpha)
                        values[name].append(logits)
                        current=trace['final_input'][q]
                        if baseline_query is None: baseline_query=current.clone()
                        torch.testing.assert_close(current,baseline_query,rtol=0,atol=0)
                        for ep in ids[~q].unique():
                            mask=(ids==ep)&~q
                            geometry.append({'stream':stream,'batch':bi,'episode':int(ep),'condition':name,
                                'pre_bn_dispersion':dispersion(trace['pre_bn'][centers][mask]),
                                'post_relu_dispersion':dispersion(trace['post_relu'][centers][mask]),
                                'U1_dispersion':dispersion(trace['U1_pre_meta'][mask]),
                                'reference_separation':float((torch.nn.functional.normalize(trace['final_label'].double(),dim=1)[1]-torch.nn.functional.normalize(trace['final_label'].double(),dim=1)[0]).norm())})
                    if batch_hash(batch)!=expected[bi]: raise ValueError('input mutated')
                    if bi%16==0:
                        print(json.dumps({'stream':stream,'batch':bi,'elapsed':time.monotonic()-started}),flush=True)
                logits={k:torch.cat(v) for k,v in values.items()}
                for key,ref in zip(['intact','suppressed'],refs):
                    torch.testing.assert_close(logits[key],ref,rtol=0,atol=0)
                for name,value in logits.items():
                    ranks,_=ranking_metrics((value[:,1].double()-value[:,0].double()).numpy(),
                        labels['local_y'].numpy(),labels['mapping'].numpy(),labels['episode_ids'].numpy(),labels['use_global'])
                    metrics.append({'stream':stream,'condition':name,**evaluate_logits(value,labels),**ranks})
                if model_digest(model.state_dict())!=rec['weights_sha256']: raise ValueError('weights mutated')
                torch.save({'logits':logits,'labels':labels},args.output/(stream+'.pt'))
                receipts.append({'stream':stream,'checkpoint':rec,'batch_sha256':expected,
                                 'native_endpoints_bitexact':True,'queries_bitexact':True})
                write_json(args.output/'metrics.json',metrics)
                write_json(args.output/'geometry.json',geometry)
                write_json(args.output/'receipts.json',receipts)
        write_json(args.output/'DONE.json',{'cells':len(metrics),'streams':len(receipts),'elapsed':time.monotonic()-started})
    except BaseException as exc:
        write_json(args.output/'FAILED.json',{'error':repr(exc)})
        raise


if __name__ == '__main__':
    main()
