from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'results/nonzero_mini_transfer'
ORDER=['ukr_rus_twitter','covid19_twitter','midterm','covid_political','election2020','ukr_rus_suspended','twibot20','cp_hk_twitter','facebook_page_reference']
LABELS=['Ukraine mini','COVID mini','Midterm','COVID Political','Election 2020','Suspended','TwiBot20','China/HK','Facebook']
def plot(path,metric,title,name):
 d=pd.read_csv(path);a=d.pivot(index='source',columns='target',values=metric).reindex(index=ORDER,columns=ORDER).to_numpy()
 fig,ax=plt.subplots(figsize=(11.8,8.4));limits={'vmin':.5,'vmax':.9} if metric=='auc' else ({'vmin':.65,'vmax':3.0} if metric=='bce' else {});im=ax.imshow(a,cmap='viridis_r' if metric!='auc' else 'viridis',**limits)
 ax.set_xticks(range(9),LABELS,rotation=35,ha='right');ax.set_yticks(range(9),LABELS)
 ax.set_xlabel('Evaluation target');ax.set_ylabel('Training source');ax.set_title(title,pad=16)
 for i in range(9):
  for j in range(9):
   c=im.cmap(im.norm(a[i,j]));lum=.2126*c[0]+.7152*c[1]+.0722*c[2]
   ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',color='black' if lum>.55 else 'white',fontsize=10)
  ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor='#ff9e00',linewidth=2))
 fig.colorbar(im,ax=ax,shrink=.8);fig.tight_layout();out=ROOT/'figures';out.mkdir(exist_ok=True)
 fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.pdf'));plt.close(fig)
if __name__=='__main__':
 plot(ROOT/'fr/feature_reconstruction_matrix.csv','scaled_cosine_error_mean','Node-only feature reconstruction • squared cosine error ↓\nDisjoint final-test nodes · 10 mask repetitions · seed 0','fr_matrix')
 for view,label in [('node','Node-only LP'),('node_neighbors','Node + 10-neighbor LP'),('node_neighbors_disjoint','Node + 10-neighbor LP · disjoint context')]:
  p=ROOT/view/'matrix.csv'
  if p.exists():
   for metric in ['auc','bce']:plot(p,metric,label+' • raw-dot '+metric.upper()+(' ↑' if metric=='auc' else ' ↓'),view+'_'+metric)
