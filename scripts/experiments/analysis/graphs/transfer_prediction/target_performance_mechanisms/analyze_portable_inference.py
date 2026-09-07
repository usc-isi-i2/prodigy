"""Validate the complete extracted-package replay against historical metric cells."""
import argparse
import hashlib
import json
import math
from pathlib import Path

TARGETS = {'covid_political', 'election2020', 'facebook_page_reference', 'twibot20', 'ukr_rus_suspended'}
SOURCES = {'cp_hk', 'ukr_rus'}
METRICS = ('roc_auc', 'accuracy', 'f1', 'nll')


def expected_conditions():
    keys = {f'topology/{q}/{s}/{d}' for d in range(3)
            for q in ('intact','removed','rewired') for s in ('intact','removed','rewired')
            if d == 0 or 'rewired' in (q,s)}
    keys |= {f'message/{c}/{r}' for c in ('actual_mean','no_message_bias','bias_only','mean_message','zero_messages')
             for r in ('query','support','both')}
    keys |= {f'dose/{d}/{r}' for d in (25,50,75) for r in range(3)}
    return keys | {'prototype/raw','prototype/encoder'}


def validate(protocol, done, rows, receipts, exported, references):
    models = {f'memberctl_{s}_lowest_sorted_s{n}' for s in SOURCES for n in range(3)}
    datasets = {f'{t}/{s}' for t in TARGETS for s in ('original','fresh')}
    panel = {(d,m) for d in datasets for m in models}
    if protocol['partial_replay'] or done['partial_replay'] or protocol['new_training']:
        raise ValueError('Completed full extracted-package inference required')
    if set(protocol['model_ids']) != models or set(protocol['dataset_ids']) != datasets or set(protocol['conditions']) != expected_conditions():
        raise ValueError('Declared full source/seed/input/condition grid differs')
    if done['cells'] != 2700 or done['model_input_cells'] != 60 or done['reference_conditions'] != 2580:
        raise ValueError('Incorrect completion counts')
    if not done['all_inputs_weights_unchanged'] or done['research_checkout_imported']:
        raise ValueError('State changed or research imports were used')
    keys = [(r['dataset_id'],r['model_id'],r['condition']) for r in rows]
    expected = {(*p,c) for p in panel for c in expected_conditions()}
    if len(rows) != 2700 or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Missing or duplicate metric cells')
    if any(not all(math.isfinite(r[m]) for m in METRICS) for r in rows):
        raise ValueError('Nonfinite metric cell')
    if len(receipts) != 60 or {(r['dataset_id'],r['model_id']) for r in receipts} != panel:
        raise ValueError('Missing or duplicate frozen-input receipts')
    counts = {'covid_political':3072,'election2020':256,'facebook_page_reference':1024,'twibot20':3072,'ukr_rus_suspended':256}
    for r in receipts:
        target = r['dataset_id'].split('/')[0]
        if (r['batches'],r['query_occurrences'],r['reference_conditions']) != (32,counts[target],43):
            raise ValueError('Incomplete batch or historical reference checks')
        if (r['query_pre_exact'],r['query_post_exact'],r['direct_role_checks']) != (288,448,37):
            raise ValueError('Missing role/query checks')
        if not r['input_and_model_unchanged'] or not r['operator_restored']:
            raise ValueError('Failed immutable-state receipt')
        if r['max_logit_error'] is None or r['max_logit_error'] > 1e-5 or r['max_metric_error'] is None or r['max_metric_error'] > 1e-6:
            raise ValueError('Historical prediction or metric discrepancy')
    if (exported['datasets'],exported['models'],exported['batches']) != (10,6,320):
        raise ValueError('Incomplete tensor export')
    audits = exported['constructor_checks']
    if len(audits) != 60 or {(f"{r['target']}/{r['stream']}",r['model_id']) for r in audits} != panel:
        raise ValueError('Incomplete constructor/input adapter parity')
    if not all(r['constructor_and_tensor_adapter_bit_exact'] for r in audits):
        raise ValueError('Changed constructor or sanitized input predictions')
    if exported['code_manifest_sha256'] != protocol['code_manifest_sha256'] or not exported['account_ids_removed'] or not exported['features_remain_private']:
        raise ValueError('Export/code identity or declared privacy scope differs')
    current = dict(zip(keys,rows))
    checked, largest = set(), 0.
    for kind, values in references.items():
        for r in values:
            if kind == 'topology': c = f"topology/{r['query_condition']}/{r['support_condition']}/{r['draw']}"
            elif kind == 'message':
                if r['condition'] == 'intact': continue
                c = f"message/{r['condition']}/{r['role']}"
            else:
                if r['step'] != 2500 or r['suppression_percent'] not in (25,50,75): continue
                c = f"dose/{r['suppression_percent']}/{r['draw']}"
            k = f"{r['target']}/{r['stream']}",r['model_id'],c
            if k in checked: raise ValueError('Duplicate historical reference cell')
            checked.add(k)
            a = current[k]
            if a['source'] != r['source'] or a['seed'] != r['seed'] or a['step'] != 2500:
                raise ValueError('Model identity changed')
            for metric in METRICS:
                delta = abs(a[metric]-r[metric])
                if not delta <= 1e-6: raise ValueError('Canonical metric mismatch: '+str(k))
                largest=max(largest,delta)
    wanted = {(*p,c) for p in panel for c in expected_conditions() if not c.startswith('prototype/')}
    if checked != wanted: raise ValueError('Canonical reference coverage incomplete')
    # Additional readouts cannot depend on which state supplied raw features.
    for dataset in datasets:
        raw = [current[(dataset,m,'prototype/raw')] for m in models]
        for metric in METRICS:
            if max(r[metric] for r in raw)-min(r[metric] for r in raw) > 1e-12:
                raise ValueError('Raw readout unexpectedly depends on checkpoint')
    return {'cells':len(rows),'canonical_reference_cells':len(checked),'additional_prototype_cells':120,
            'saved_final_checkpoints':6,'targets':5,'streams':2,'constructor_adapter_exact_checks':60,
            'query_pre_exact_checks':sum(r['query_pre_exact'] for r in receipts),
            'query_post_exact_checks':sum(r['query_post_exact'] for r in receipts),
            'direct_role_checks':sum(r['direct_role_checks'] for r in receipts),
            'max_logit_error':max(r['max_logit_error'] for r in receipts),
            'max_canonical_metric_error':largest,'research_checkout_imported':False,
            'code_manifest_sha256':protocol['code_manifest_sha256'],'environment':protocol['environment'],
            'elapsed_seconds':done['elapsed_seconds'],
            'scope':'Complete final-checkpoint role/topology/message/dose replay from extracted code; not training, raw graph construction, the nine-source historical map or all saved steps.'}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,required=True,help='Folder containing full/ and export_receipt.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    leaf=Path(__file__).resolve().parent/'data'
    read=lambda path:json.loads(path.read_text())
    files={n:a.input/'full'/(n+'.json') for n in ('protocol','DONE','metrics','receipts')}
    files['export_receipt']=a.input/'export_receipt.json'
    refs={k:read(leaf/f/'metrics.json') for k,f in [('topology','role_topology_20260907'),('message','message_content_20260907'),('dose','support_dose_20260907')]}
    result=validate(*[read(files[n]) for n in ('protocol','DONE','metrics','receipts','export_receipt')],refs)
    result['source_sha256']={n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
