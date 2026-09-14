"""Same-year negative-generalization probe; positives deliberately remain training examples."""
import argparse, importlib.util, json, subprocess, time
from pathlib import Path
import numpy as np
import torch
import wandb


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--original',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    assert a.device in ['cuda:0','cuda:1','cuda:2','cuda:3']
    root=Path(__file__).resolve().parents[5]
    spec=importlib.util.spec_from_file_location('probe_joint',root/'scripts/experiments/setup/ogbl_collab_compact_joint/run.py'); j=importlib.util.module_from_spec(spec); spec.loader.exec_module(j)
    a.out.mkdir(parents=True,exist_ok=False)
    cfg=dict(revision=j.revision(),source=str(a.source),original=str(a.original),device=a.device,year=2017,checkpoints=['best','final'],seeds=[0,1,2],test_access=False,training=False,question='Does negative generalization degrade holding positives, graph, and year fixed?',qualification='Training positives in both comparisons; unseen means absent from this model training-negative set, not pristine research holdout.')
    j.save(a.out/'protocol.json',cfg); w=wandb.init(project='ogbl-collab-compact',group='fresh-diagnosis-v1',name='same-year-probe',mode='offline',dir=str(a.out),config=cfg)
    pm=j.read(a.source/'prepared.json'); orig=j.read(a.original/'prepared.json')
    for name in ['node_features.npy','standardization.npz']:
        assert j.shared.sha256_file(a.original/name)==pm['files'][str(a.original/name)]
    probe_path=a.original/'year2017.npz'
    assert j.shared.sha256_file(probe_path)==orig['files']['year2017.npz']
    trpath=a.source/'train2017.npz'; assert j.shared.sha256_file(trpath)==pm['files'][str(trpath)]
    tr=np.load(trpath); old=np.load(probe_path); np.testing.assert_array_equal(tr['graph_edges'],old['graph_edges'])
    x=torch.as_tensor(np.load(a.original/'node_features.npy'),device=a.device); ms=np.load(a.original/'standardization.npz')
    keys=j.gate.keys; trainkeys=keys(tr['neg'],len(x)); oldkeys=keys(old['neg'],len(x))
    keep=~np.isin(oldkeys,trainkeys)
    assert not np.intersect1d(keys(tr['pos'],len(x)),oldkeys).size
    probe=dict(pos=tr['pos'],psymmetric=tr['psymmetric'],neg=old['neg'][keep],nsymmetric=old['nsymmetric'][keep])
    tp=j.panel_tensors(tr,ms['mean'],ms['std'],a.device); up=j.panel_tensors(probe,ms['mean'],ms['std'],a.device)
    adj=j.adjacency(tr['graph_edges'],len(x),a.device); torch.set_num_threads(4)
    rows=[]; start=time.monotonic(); warm=j.gate.warm_positive_mask(tr['pos'],tr['graph_edges'],len(x))
    for seed in range(3):
        record=j.read(a.source/f'select{seed}/results.json')
        for label,hashkey in [('best','best_checkpoint_sha256'),('final','final_sha256')]:
            path=a.source/f'select{seed}/{label}.pt'; assert j.shared.sha256_file(path)==record[hashkey]
            state=torch.load(path,map_location=a.device,weights_only=False); model=j.Model('joint').to(a.device); model.load_state_dict(state['model']); model.eval()
            with torch.no_grad():
                z=model.encode(x,adj); positive=j.scores(model,z,tp['p']).cpu().numpy(); seen=j.scores(model,z,tp['n']).cpu().numpy(); unseen=j.scores(model,z,up['n']).cpu().numpy()
            def summary(negative):
                threshold=float(np.partition(negative,-50)[-50]); hits=positive>threshold
                return dict(cutoff=threshold,hits=float(hits.mean()),warm_hits=float(hits[warm].mean()),cold_hits=float(hits[~warm].mean()),negative_quantiles=np.quantile(negative,[.5,.99,.999,.9995]).tolist())
            row=dict(seed=seed,checkpoint=label,step=state['step'],seen=summary(seen),unseen=summary(unseen),checkpoint_sha256=record[hashkey],positive_quantiles=np.quantile(positive,[.1,.5,.9]).tolist()); rows.append(row)
            np.savez_compressed(a.out/f'seed{seed}_{label}.npz',p=positive,seen=seen,unseen=unseen)
            print(json.dumps(row),flush=True); del model,z
    out=dict(**cfg,rows=rows,positive_count=len(tr['pos']),warm_positive_count=int(warm.sum()),seen_negative_count=len(trainkeys),unseen_negative_count=int(keep.sum()),removed_overlap=int((~keep).sum()),elapsed_seconds=time.monotonic()-start,wandb_directory=w.dir,probe_panel_sha256=j.shared.sha256_file(probe_path),train_panel_sha256=j.shared.sha256_file(trpath))
    j.save(a.out/'results.json',out); w.log({'probe/complete':1}); w.finish()
if __name__=='__main__': main()
