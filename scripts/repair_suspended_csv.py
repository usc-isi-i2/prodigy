"""Repair two unquoted multiline bios, preserve row order, reuse content-hashed embeddings."""
import csv,hashlib,io,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,'/dataMeR1/phil/gfm/prodigy')
from scripts.bio_embeddings.preprocessing import normalize_bio_text,bio_hash
root=Path('/dataMeR1/phil/data');src=root/'social_llm_data/ukr_rus_suspended';out=Path('/dataMeR1/phil/gfm/mixture-scaling/results/suspended_csv_repair_20260911');out.mkdir(parents=True,exist_ok=False)
raw=(src/'user_data.csv').read_bytes();reader=csv.reader(io.StringIO(raw.decode()));header=next(reader);records=list(reader);fixed=[];repairs=[];i=0
while i<len(records):
 row=records[i]
 if len(row)==14:fixed.append(row);i+=1;continue
 assert len(row)==2 and row[0] in ('0','1'),(i,len(row))
 start=i;parts=[row[1]];i+=1
 while len(records[i])==1:parts.append(records[i][0]);i+=1
 assert len(records[i])==13,(i,len(records[i]));parts.append(records[i][0]);fixed.append([row[0],'\n'.join(parts),*records[i][1:]]);repairs.append({'first_csv_record':start,'last_csv_record':i,'node_row':len(fixed)-1});i+=1
G=pickle.load((src/'graph.pickle').open('rb'));ids=np.load(src/'user_ids.npy',allow_pickle=True)
assert len(fixed)==len(ids)==len(G)==56440 and set(G.nodes)==set(range(len(fixed)))
assert len(repairs)==2
for row in fixed:
 for j,v in enumerate(row):
  if j!=1:float(v)
with (out/'user_data.corrected.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(header);w.writerows(fixed)
a=pd.read_csv(out/'user_data.corrected.csv',engine='c');b=pd.read_csv(out/'user_data.corrected.csv',engine='python');pd.testing.assert_frame_equal(a,b);assert len(a)==56440
old=torch.load(root/'ukr_rus_suspended/embeddings/user_bio_embeddings_gte_multilingual_base.pt',map_location='cpu',weights_only=False)
lookup={}
for i,h in enumerate(old['bio_hashes']):
 if h:lookup.setdefault(h,i)
texts=[normalize_bio_text(v) for v in a.profile.fillna('')];hashes=[bio_hash(t) if t else '' for t in texts];missing={h:t for h,t in zip(hashes,texts) if h and h not in lookup}
print('Verified rows',len(a),'repairs',repairs,'missing unique embeddings',len(missing),flush=True)
newvectors={}
if missing:
 from scripts.tweet_embeddings.model_backend import load_model_for_worker,encode_texts
 cfg={'model':old['model'],'revision':old['revision'],'embedding_dim':768,'batch_size':32,'max_seq_length':512,'fp16':True,'cpu':False,'cache_folder':''}
 model,device=load_model_for_worker(0,cfg);encoded=encode_texts(model,list(missing.values()),cfg);newvectors=dict(zip(missing,encoded))
x=torch.zeros((len(a),768),dtype=old['meanpool'].dtype)
for i,h in enumerate(hashes):
 if h in lookup:x[i]=old['meanpool'][lookup[h]]
 elif h:x[i]=torch.as_tensor(newvectors[h])
emb={**old,'user_ids':np.arange(len(a)),'handles':[None]*len(a),'meanpool':x,'counts':np.array([bool(h) for h in hashes],dtype=np.int64),'bio_hashes':hashes}
torch.save(emb,out/'embeddings.corrected.pt')
report={'source_sha256':hashlib.sha256(raw).hexdigest(),'old_rows':len(old['user_ids']),'corrected_rows':len(a),'edges':G.number_of_edges(),'repairs':repairs,'missing_unique_reencoded':len(missing),'zero_vectors':int((x==0).all(1).sum()),'parser_agreement':True,'node_ids_match':True,'status':'candidate_not_promoted'}
(out/'repair_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
