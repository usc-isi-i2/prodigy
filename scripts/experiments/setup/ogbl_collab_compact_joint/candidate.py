"""Validation-selected candidate with fresh negatives and fixed final refit."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import run as joint
from train2018 import grid
from fusion import cutoff, hits, maximum, transform


def fresh_negatives(positive, validation_negative, n, year):
    # Validation exclusions are allowed development information. No test input.
    candidates = joint.gate.negatives(positive,n,20260000+year,count=100100)
    blocked = joint.gate.keys(validation_negative,n)
    keep = ~np.isin(joint.gate.keys(candidates,n),blocked)
    result = candidates[keep][:100000]
    assert len(result)==100000
    return result


def contract():
    return dict(name='candidate_fresh_v1',revision=joint.revision(),seeds=[0,1,2],
        select_train_year=2017,select_validation_year=2018,refit_train_year=2018,
        selection_steps=2000,selection_interval=50,final_year=2019,
        negative_sampling='uniform unique target-year nonpositives, seed20260000+year; excludes validation negatives; does not inspect test identities',
        selection='per seed highest2018 fusion Hits; earliest step then declared base/alpha order',
        final='fresh initialization, exactly validation-selected update count; alpha and both normalizers frozen from validation; one test evaluation after final checkpoint saved',
        unchanged='496385 neural weights;2015 AA feature calibration;2016 standardization;balanced BCE,Adam.001 wd.0001 clip5,2048pos/1024uniform/1024hard,top2048 refreshed10updates',
        total_inference_scalars=496448,test_selection=False,prior_test_exploration_disclosed=True,
        eligibility='candidate aligned with documented protocol; maintainer acceptance not established',
        stop='three selection and three fixed refit cells; no test-tuned expansion')


def verify(args):
    cfg=joint.read(args.out/'protocol.json'); assert cfg==contract()
    meta=joint.read(args.out/'prepared.json')
    for path,sha in meta['files'].items():
        assert joint.shared.sha256_file(Path(path))==sha
    return cfg,meta


def prepare(args):
    args.out.mkdir(parents=True,exist_ok=False)
    cfg=contract(); joint.save(args.out/'protocol.json',cfg)
    run=tracking(args.out,'prepare',cfg)
    prepared=joint.read(args.runtime/'prepared.json')
    graph,split,_,version=joint.shared.load_official_dataset(args.dataset_root)
    identity=joint.shared.audit_dataset(graph,split,version)
    assert identity['split_fingerprint']==joint.gate.FINGERPRINT
    del split['test'] # Full-split metadata audit only; no test affects preparation.
    assert subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()==joint.gate.UPSTREAM
    source=args.upstream/'aa_dc.py'
    assert source.read_bytes()==subprocess.check_output(['git','-C',str(args.upstream),'show','HEAD:aa_dc.py'])
    aa=joint.module(source,'candidate_aa')
    files={}
    for name in ('node_features.npy','standardization.npz'):
        path=args.runtime/name
        assert joint.shared.sha256_file(path)==prepared['files'][name]
        files[str(path)]=prepared['files'][name]
    assessment=args.runtime/'assessment/year2018.npz'
    expected=joint.read(args.runtime/'assessment/results.json')['panel_sha256']
    assert joint.shared.sha256_file(assessment)==expected
    files[str(assessment)]=expected
    metadata=[]
    for year in (2017,2018):
        positive=split['valid']['edge'] if year==2018 else split['train']['edge'][split['train']['year'].flatten()==2017]
        neg=fresh_negatives(positive,split['valid']['edge_neg'],int(graph['num_nodes']),year)
        panel,meta,_=joint.gate.make_year(aa,joint.shared,graph,split,year,prepared['calibration'],
            symmetric_features=True,negative_edges=neg)
        panel['graph_edges']=joint.graph_edges(graph,split,year)
        path=args.out/f'train{year}.npz'; np.savez_compressed(path,**panel)
        files[str(path)]=joint.shared.sha256_file(path)
        meta['validation_negative_overlap']=int(len(np.intersect1d(joint.gate.keys(neg,int(graph['num_nodes'])),joint.gate.keys(split['valid']['edge_neg'],int(graph['num_nodes'])))))
        assert meta['validation_negative_overlap']==0
        metadata.append(meta)
    joint.save(args.out/'prepared.json',dict(files=files,metadata=metadata,identity=identity,
        producer_revision=prepared['revision'],wandb_directory=run.dir))
    run.finish()


def tracking(out,name,cfg):
    import wandb
    return wandb.init(project='ogbl-collab-compact-joint',group='candidate_fresh_v1',name=name,
                      mode='offline',dir=str(out),config=cfg)


def freeze(args):
    cfg,_=verify(args)
    assert not (args.out/'frozen.json').exists()
    selections=[]
    for seed in range(3):
        p=args.out/f'select{seed}'
        r=joint.read(p/'results.json')
        assert r['complete'] and r['revision']==cfg['revision'] and not r['test_scored']
        assert joint.shared.sha256_file(p/'history.json')==r['history_sha256']
        assert joint.shared.sha256_file(p/'best.pt')==r['best_checkpoint_sha256']
        selections.append(r)
    joint.save(args.out/'frozen.json',dict(selections=selections,revision=cfg['revision'],test_scored=False))


def train(args):
    cfg,meta=verify(args)
    refit=args.phase=='refit'
    frozen=joint.read(args.out/'frozen.json') if refit else None
    selected=frozen['selections'][args.seed]['best'] if refit else None
    steps=selected['step'] if refit else 2000
    out=args.out/f'{args.phase}{args.seed}'; out.mkdir(exist_ok=False)
    run=tracking(out,out.name,cfg)
    torch.set_num_threads(8); torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    generator=torch.Generator().manual_seed(args.seed)
    tr=np.load(args.out/f"train{2018 if refit else 2017}.npz")
    ms=np.load(args.runtime/'standardization.npz')
    x=torch.as_tensor(np.load(args.runtime/'node_features.npy'),device=args.device)
    tp=joint.panel_tensors(tr,ms['mean'],ms['std'],args.device)
    adj=joint.adjacency(tr['graph_edges'],len(x),args.device)
    if not refit:
        va=np.load(args.runtime/'assessment/year2018.npz')
        vp=joint.panel_tensors(va,ms['mean'],ms['std'],args.device)
        vadj=joint.adjacency(va['graph_edges'],len(x),args.device)
    model=joint.Model('joint').to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    history=[]; best=None; started=time.monotonic()
    for step in range(1,steps+1):
        if (step-1)%10==0:
            with torch.no_grad():
                model.eval(); z=model.encode(x,adj)
                hard=joint.scores(model,z,tp['n']).topk(2048).indices.cpu(); del z
        pi,ni,_=joint.sampled_indices(generator,len(tp['p'][0]),len(tp['n'][0]),hard)
        model.train(); optimizer.zero_grad(set_to_none=True); z=model.encode(x,adj)
        values=[]
        for side,index in [('p',pi),('n',ni)]:
            index=index.to(args.device); edges,features=tp[side]
            values.append(model.score(z,edges[index],features[index]))
        loss=nn.functional.binary_cross_entropy_with_logits(torch.cat(values),torch.cat([torch.ones_like(values[0]),torch.zeros_like(values[1])]))
        assert torch.isfinite(loss); loss.backward()
        grad=nn.utils.clip_grad_norm_(model.parameters(),5.); assert torch.isfinite(grad)
        optimizer.step(); del z,values
        if step%10==0: run.log({'step':step,'train/loss':float(loss)})
        if step%50==0:
            row=dict(step=step,loss=float(loss),elapsed_seconds=time.monotonic()-started)
            if not refit:
                scores=joint.evaluate(model,x,vadj,vp)
                rows=grid(va,scores); winner=max(rows,key=lambda r:r['hits'])
                row.update(grid=rows,joint_hits=hits(**scores))
                if best is None or winner['hits']>best['hits']:
                    prefix='official_b' if winner['base']=='validation2018' else 'b'
                    best=dict(step=step,**winner,scale=cutoff(va[prefix+'n']),offset=cutoff(scores['n']))
                    torch.save(dict(model=model.state_dict(),step=step,seed=args.seed,revision=cfg['revision']),out/'best.pt')
                    np.savez_compressed(out/'best_scores.npz',**scores)
                run.log({'step':step,'validation/joint_hits':row['joint_hits'],'validation/best_fusion':best['hits']})
            history.append(row); joint.save(out/'history.json',history)
            print(json.dumps(dict(phase=args.phase,seed=args.seed,step=step,best=best)),flush=True)
    torch.save(dict(model=model.state_dict(),step=steps,seed=args.seed,revision=cfg['revision']),out/'final.pt')
    result=dict(complete=True,revision=cfg['revision'],seed=args.seed,steps=steps,best=best,
        test_scored=False,history_sha256=joint.shared.sha256_file(out/'history.json'),
        final_sha256=joint.shared.sha256_file(out/'final.pt'),elapsed_seconds=time.monotonic()-started,wandb_directory=run.dir)
    if not refit:
        state=torch.load(out/'best.pt',map_location=args.device,weights_only=False)
        model.load_state_dict(state['model'])
        replay=joint.evaluate(model,x,vadj,vp)
        saved=np.load(out/'best_scores.npz')
        for side in ('p','n'): np.testing.assert_array_equal(replay[side],saved[side])
        result.update(best_checkpoint_sha256=joint.shared.sha256_file(out/'best.pt'),
            best_score_sha256=joint.shared.sha256_file(out/'best_scores.npz'),exact_selected_replay=True)
    else:
        # No test access until training has ended and its checkpoint is saved.
        testmeta=joint.read(args.test_runtime/'results.json')
        testpath=args.test_runtime/'year2019.npz'
        assert joint.shared.sha256_file(testpath)==testmeta['panel_sha256']
        te=np.load(testpath)
        keys=joint.gate.keys
        overlap=int(len(np.intersect1d(keys(tr['neg'],len(x)),keys(te['neg'],len(x)))))
        assert overlap==0, 'Unexpected accidental test-negative overlap; do not resample based on test'
        testp=joint.panel_tensors(te,ms['mean'],ms['std'],args.device)
        testadj=joint.adjacency(te['graph_edges'],len(x),args.device)
        scores=joint.evaluate(model,x,testadj,testp)
        np.savez_compressed(out/'test_scores.npz',**scores)
        prefix='official_b' if selected['base']=='validation2018' else 'b'
        z=transform({'bp':te[prefix+'p'],'bn':te[prefix+'n']},scores,selected['scale'],selected['offset'])
        f=maximum(z,selected['alpha'])
        from ogb.linkproppred import Evaluator
        evaluator=Evaluator(name='ogbl-collab'); evaluator.K=50
        score=hits(**f)
        assert score==evaluator.eval({'y_pred_pos':f['p'],'y_pred_neg':f['n']})['hits@50']
        result.update(test_scored=True,test_hits=score,joint_test_hits=hits(**scores),frozen_settings=selected,
            test_negative_training_overlap=overlap,total_inference_scalars=496448,
            test_panel_sha256=testmeta['panel_sha256'],test_score_sha256=joint.shared.sha256_file(out/'test_scores.npz'),
            frozen_sha256=joint.shared.sha256_file(args.out/'frozen.json'))
        run.log({'test/fusion_hits':score,'test/joint_hits':result['joint_test_hits']})
    joint.save(out/'results.json',result); run.finish(); print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','select','freeze','refit'])
    p.add_argument('--runtime',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--test-runtime',type=Path,default=Path('/dataMeR1/phil/gfm/ogbl_collab_compact_joint/official_test_fusion_v1'))
    p.add_argument('--dataset-root',type=Path,default=Path('/dataMeR1/phil/data/ogb'))
    p.add_argument('--upstream',type=Path,default=Path('/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204/upstream'))
    p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    p.add_argument('--device',choices=['cuda:0','cuda:1','cuda:2','cuda:3'],default='cuda:0')
    p.add_argument('--dry-run',action='store_true'); args=p.parse_args()
    if args.dry_run: print(json.dumps(contract(),indent=2)); return
    {'prepare':prepare,'select':train,'freeze':freeze,'refit':train}[args.phase](args)


if __name__=='__main__': main()
