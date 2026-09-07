"""Read-only audit of three fixed examples on Tucker; no model forwards."""
import pathlib,json,pickle,torch,hashlib
torch.set_num_threads(1)
base=pathlib.Path('/dataMeR1/phil/gfm/prodigy-publickg-paired-state/log')
root=base/'publickg_readout_fresh_20260907';ds=pathlib.Path('/dataMeR1/phil/data/prodigy_public_original/FB15K-237')
entities={int(v):k for k,v in json.loads((ds/'entity2id.json').read_text()).items()}
rels=json.loads((ds/'relation2id.json').read_text())
names=pickle.loads((ds/'mid2name_dict.pkl').read_bytes())
texts,emb=torch.load(ds/'preproc_text_feats/text_feats_sentence-transformers_all-mpnet-base-v2_.pkl',map_location='cpu',weights_only=False)
indices={s:i for i,s in enumerate(texts)};rnames=list(rels);remb=torch.stack([emb[indices[s]] for s in rnames])
index=json.loads((root/'paired_episodes/index.json').read_text())['records']
cases=[]
for ep,q in [(0,28),(0,52),(1,15)]:
 p=root/'paired_episodes'/index[ep]['file'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==index[ep]['sha256']
 a=torch.load(p,map_location='cpu',weights_only=False)['native_capture']['input']
 n=a[2].shape[0];qm=a[5].reshape(n,-1)[:,0]==1;qi=torch.where(qm)[0]
 cls=[]
 for x in a[1]:
  match=torch.where((remb==x).all(1))[0];assert len(match)==1;cls.append(rnames[int(match[0])])
 cross=[torch.load(base/'publickg_crossover_20260907'/c/f'episode_{ep:05d}.pt',map_location='cpu',weights_only=False) for c in ('E2000I2000','E2000I8000','E8000I2000','E8000I8000')]
 assert torch.equal(a[2][qi],cross[0]['y_true_onehot'])
 pred=[int(c['logits'][q].argmax()) for c in cross];truth=int(a[2][qi[q]].argmax())
 def pair(i):
  ids=a[0].center_node_idx[int(i)]
  return [{'id':int(j),'entity':entities[int(j)],'alias':names.get(entities[int(j)],entities[int(j)])} for j in ids]
 support_idx=a[6][0,::2];support_cls=a[6][0,1::2]-n
 assert len(support_idx)==60 and not qm[support_idx].any()
 supports={}
 for c in sorted(set(pred+[truth])):
  si=support_idx[support_cls==c];assert len(si)==3
  assert (a[2][si].argmax(1)==c).all()
  supports[cls[c]]=[pair(i) for i in si]
 cases.append({'episode':ep,'query':q,'data_point_index':int(qi[q]),'pair':pair(qi[q]),'truth':cls[truth],'predictions':[cls[c] for c in pred],'supports':supports,'label_match_max_error':0,'source_sha256':index[ep]['sha256']})
print(json.dumps({'condition_order':['E2000I2000','E2000I8000','E8000I2000','E8000I8000'],'cases':cases,'scope':'First three preselected crossover failures; dataset aliases, not independently verified names; relation mapping via exact embedding equality; query mask order verified against saved truth'},ensure_ascii=False))
