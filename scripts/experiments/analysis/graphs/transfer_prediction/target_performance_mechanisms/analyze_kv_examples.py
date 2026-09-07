import torch,json,numpy as np
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description="Read saved K/V logits; report paired transitions without model forwards.")
parser.add_argument("--root", type=Path, required=True)
root=parser.parse_args().root
out=[]
for stream in ["original","fresh"]:
 a=torch.load(root/"private_predictions"/(stream+".pt"),map_location="cpu",weights_only=False)
 y=a["labels"]["local_y"].numpy(); ids=a["labels"]["episode_ids"].numpy()
 scores={k:(v[:,1].double()-v[:,0].double()).numpy() for k,v in a["logits"].items()}
 correct={k:v.argmax(1).numpy()==y for k,v in a["logits"].items()}
 rows=[]; signs=[]
 for ep in np.unique(ids):
  ii=np.where(ids==ep)[0]; pos=ii[y[ii]==1]; neg=ii[y[ii]==0]
  credits={k:(np.sign(v[pos,None]-v[None,neg])+1)/2 for k,v in scores.items()}
  auc={k:float(v.mean()) for k,v in credits.items()}
  di=credits["values_only"]-credits["intact"]; dk=credits["keys_only"]-credits["intact"]
  signs.append((di.ravel(),dk.ravel()))
  rows.append(dict(episode=int(ep),delta_values=auc["values_only"]-auc["intact"],delta_keys=auc["keys_only"]-auc["intact"],aucs=auc))
 dv=np.array([r["delta_values"] for r in rows]); dk=np.array([r["delta_keys"] for r in rows])
 vp=np.concatenate([r[0] for r in signs]); kp=np.concatenate([r[1] for r in signs])
 changed_v=correct["values_only"]!=correct["intact"]; changed_k=correct["keys_only"]!=correct["intact"]
 loss=[r for r in rows if r["delta_values"]<0]; representative=sorted(loss,key=lambda r:r["delta_values"])[len(loss)//2]
 ep=representative["episode"]; ii=np.where(ids==ep)[0]
 cand=ii[(correct["intact"][ii]) & (~correct["values_only"][ii])]
 examples=[dict(query_offset=int(i),episode=ep,local_label=int(y[i]),margins={k:float(v[i]) for k,v in scores.items()}) for i in cand[:3]]
 out.append(dict(stream=stream,episode_count=len(rows),value_harmed_episodes=int((dv<0).sum()),value_helped_episodes=int((dv>0).sum()),value_tied_episodes=int((dv==0).sum()),delta_auc_quantiles_points=(100*np.quantile(dv,[0,.25,.5,.75,1])).tolist(),value_mean_points=float(100*dv.mean()),value_mean_without_worst_episode=float(100*np.delete(dv,dv.argmin()).mean()),key_helped_episodes=int((dk>0).sum()),key_harmed_episodes=int((dk<0).sum()),pair_counts={"total":len(vp),"value_hurt":int((vp<0).sum()),"value_help":int((vp>0).sum()),"key_hurt":int((kp<0).sum()),"key_help":int((kp>0).sum()),"value_hurt_key_help":int(((vp<0)&(kp>0)).sum()),"value_hurt_key_unchanged":int(((vp<0)&(kp==0)).sum())},accuracy_transitions={k:{"lost_correct":int((correct["intact"]&~correct[k]).sum()),"gained_correct":int((~correct["intact"]&correct[k]).sum())} for k in ["keys_only","values_only"]},changed_prediction_overlap={"value":int(changed_v.sum()),"key":int(changed_k.sum()),"both":int((changed_v&changed_k).sum())},median_harmed_episode=representative,examples=examples))
print(json.dumps(out,indent=2))
