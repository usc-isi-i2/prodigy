from pathlib import Path
import pandas as pd,numpy as np
P=Path('/tmp/mlp-gradient-diagnostic');gs=[];us=[]
for case in (0,1):
 for name,frames in [('gradients',gs),('updates',us)]:
  d=pd.read_csv(P/f'case{case}'/(name+'.csv'));d['case']=case;frames.append(d)
g=pd.concat(gs);u=pd.concat(us)
g['conflict']=g.cosine<0;g['norm_ratio']=g.norm_right/g.norm_left
summary=g.groupby(['case','state','group','left','right']).agg(cosine=('cosine','mean'),conflict_fraction=('conflict','mean'),norm_left=('norm_left','mean'),norm_right=('norm_right','mean'),norm_ratio=('norm_ratio','mean')).reset_index()
u['bce_harmed']=u.bce_delta>0;u['auc_harmed']=u.auc_delta_pp<0
updates=u.groupby(['case','state','action','validation_source']).agg(bce_delta=('bce_delta','mean'),bce_sd=('bce_delta','std'),bce_harmed=('bce_harmed','sum'),auc_delta_pp=('auc_delta_pp','mean'),auc_harmed=('auc_harmed','sum'),predicted_bce=('first_order_bce_delta','mean'),n=('trial','count')).reset_index()
print('RAW A/B GRADIENT ALIGNMENT')
print(summary[(summary.left=='train_A')&(summary.right=='train_B')].round(5).to_string(index=False))
print('ALIGNMENT WITH VALIDATION (ALL PARAMETERS)')
print(summary[(summary.group=='all')&(summary.right.str.startswith('val'))].round(5).to_string(index=False))
print('UPDATE EFFECTS')
print(updates.round(7).to_string(index=False))
summary.to_csv(P/'gradient_summary.csv',index=False);updates.to_csv(P/'update_summary.csv',index=False)
