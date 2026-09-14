from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/tmp/mlp-gradient-diagnostic');g=pd.read_csv(p/'gradient_summary.csv');u=pd.read_csv(p/'step_scale_details.csv')
fig,axs=plt.subplots(2,3,figsize=(16,9));titles=['Ukraine + Facebook','Suspended + COVID political']
for case in (0,1):
 ax=axs[case,0];z=g[(g.case==case)&(g.group=='all')&(g.left=='train_A')&(g.right=='train_B')].set_index('state').reindex(['initial','rank0_best','rank0_final'])
 ax.plot(range(3),z.cosine,marker='o',color='#405777',lw=2)
 for i,v in enumerate(z.cosine):ax.text(i,v+.09,f'{v:+.2f}',ha='center',fontsize=11)
 ax.axhline(0,color='#888',lw=.8);ax.set_ylim(-1,1);ax.set_xticks(range(3),['Initial','Control best','Control final']);ax.set_ylabel(titles[case]+'\nA/B training-gradient cosine',fontsize=11)
 for j in [1,2]:
  ax=axs[case,j];state='initial' if j==1 else 'rank0_best';d=u[(u.case==case)&(u.state==state)&(u.action=='B_task')].copy()
  for source,color in [('A','#2478a2'),('B','#cd792f')]:
   q=d[d.validation_source==source].copy();q['value']=100*q.bce_delta/q.bce_before if j==1 else q.auc_delta_pp
   q=q.groupby('update_scale').value.mean();ax.plot(q.index,q.values,marker='o',lw=2,color=color,label=f'Validation {source}')
  ax.axhline(0,color='#888',lw=.8);ax.set_xticks([.1,.25,.5,1],['10%','25%','50%','100%']);ax.set_xlabel('Fraction of the same AdamW update')
  ax.set_ylabel('Validation BCE change (%)\nLower is better' if j==1 else 'Validation AUC change (pp)\nHigher is better')
  ax.set_ylim((-30,25) if j==1 else (-.055,.095));ax.legend(frameon=False,fontsize=9)
 for ax in axs[case]:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
axs[0,0].set_title('Gradient conflict changes over training',fontsize=12)
axs[0,1].set_title('First B update: a full step overshoots A',fontsize=12)
axs[0,2].set_title('B update at selected control checkpoint',fontsize=12)
fig.suptitle('Gradient alignment alone does not explain transfer',fontsize=20,fontweight='bold',y=.98)
fig.text(.5,.932,'Eight training-batch probes per saved state · fixed A/B validation pairs · weights and AdamW state restored before every action',ha='center',fontsize=10)
fig.subplots_adjust(top=.86,bottom=.13,left=.08,right=.99,hspace=.4,wspace=.34)
fig.text(.5,.025,'A is the starting singleton; B is the added graph. Full update uses the training learning rate (0.0005) and gradient clipping.\nScaled probes change only the parameter-step magnitude. These are local effects, not completed lower-learning-rate training runs.',ha='center',fontsize=10,color='#555')
fig.savefig(p/'gradient_update_diagnostic.png',dpi=180,facecolor='white');fig.savefig(p/'gradient_update_diagnostic.svg',facecolor='white')
