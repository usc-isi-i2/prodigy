"""Read-only Tucker diagnostic; no forwards. Fixed paths pin the audited runs."""
import json,pathlib,hashlib,torch,collections
torch.set_num_threads(1)
base=pathlib.Path('/dataMeR1/phil/gfm/prodigy-publickg-paired-state/log')
cr=base/'publickg_crossover_20260907';tr=base/'publickg_trajectory_20260907'
cs=json.loads((cr/'summary.json').read_text());ts=json.loads((tr/'summary.json').read_text())
assert json.loads((cr/'execution_status.json').read_text())['status']=='complete'
assert json.loads((tr/'execution_status.json').read_text())['status']=='complete'
refs={str(cr/r['file']):r['sha256'] for r in cs['receipts']}
refs.update({str(tr/r['file']):r['sha256'] for r in ts['receipts'] if r['step'] in (2000,8000)})
assert len(refs)==768
def read(p):
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==refs[str(p)]
 return torch.load(p,map_location='cpu',weights_only=False)
conditions=['E2000I2000','E2000I8000','E8000I2000','E8000I8000']
patterns=collections.Counter();stablepatterns=collections.Counter();allcorrect=[];ridge=[];examples=[]
for ep in range(128):
 cells=[read(cr/c/f'episode_{ep:05d}.pt') for c in conditions]
 temporal=[read(tr/str(s)/f'episode_{ep:05d}.pt') for s in (2000,8000)]
 y=cells[0]['y_true_onehot'];source=cells[0]['source_sha256'];truth=y.argmax(1)
 assert y.shape==(80,20)
 for cell in cells+temporal:
  assert cell['source_sha256']==source and torch.equal(cell['y_true_onehot'],y)
 for ci,ti in ((0,0),(3,1)):
  torch.testing.assert_close(cells[ci]['logits'],temporal[ti]['native_logits'],atol=1e-4,rtol=0)
  assert torch.equal(cells[ci]['logits'].argmax(1),temporal[ti]['native_logits'].argmax(1))
 pred=torch.stack([x['logits'].argmax(1) for x in cells],1)
 corr=pred==truth[:,None]
 rp=torch.stack([x['centered_logits'].argmax(1) for x in temporal],1)
 rc=rp==truth[:,None]
 allcorrect.append(corr);ridge.append(rc)
 for q in range(80):
  pattern=''.join(str(int(x)) for x in corr[q]);patterns[pattern]+=1
  if bool(rc[q].all()):stablepatterns[pattern]+=1
  if pattern=='1100' and bool(rc[q].all()) and len(examples)<3:
   examples.append({'episode':ep,'query':q,'truth_local':int(truth[q]),'predictions':pred[q].tolist(),'ridge_predictions':rp[q].tolist()})
a=torch.cat(allcorrect);r=torch.cat(ridge)
def count(mask):return int(mask.sum())
damage=a[:,0]&~a[:,2]
repair=damage&a[:,3]
out={'query_occurrences':len(a),'condition_order':conditions,'correctness_patterns':dict(sorted(patterns.items())),'ridge_correct_both_patterns':dict(sorted(stablepatterns.items())),'correct_counts':{c:count(a[:,j]) for j,c in enumerate(conditions)},'early_inference_encoder_damage':count(damage),'damage_repaired_by_late_inference':count(repair),'damage_remaining_with_late_inference':count(damage&~a[:,3]),'encoder_damage_under_both_inference_modules':count(a[:,0]&a[:,1]&~a[:,2]&~a[:,3]),'same_damage_with_ridge_correct_both':count(a[:,0]&a[:,1]&~a[:,2]&~a[:,3]&r.all(1)),'late_inference_late_encoder_corrections':count(~a[:,2]&a[:,3]),'late_inference_late_encoder_corruptions':count(a[:,2]&~a[:,3]),'first_three_both_inference_damage_ridge_correct':examples,'verification':{'hashed_files':len(refs),'episodes':128,'query_labels_source_hashes_match':True,'diagonal_argmax_exact':True,'diagonal_logit_atol':1e-4},'summary_sha256':{'crossover':hashlib.sha256((cr/'summary.json').read_bytes()).hexdigest(),'trajectory':hashlib.sha256((tr/'summary.json').read_bytes()).hexdigest()},'scope':'Post-hoc saved-prediction accounting, not independent accounts, causal training-gradient mediation, or semantic example verification'}
print(json.dumps(out))
