"""Render seven-graph joint feature MMD and its sampling-noise baselines."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'data/feature_joint_mmd_seven.json').read_text())
a=np.array(r['mean']);labels=['UKR/RUS','COVID','Midterm','Suspended','TwiBot20','CP/HK','Facebook']
fig,ax=plt.subplots(figsize=(10,8.5))
im=ax.imshow(a,cmap='viridis',vmin=0,vmax=a.max())
ax.set_xticks(range(7),labels,rotation=45,ha='right');ax.set_yticks(range(7),labels)
for i in range(7):
 for j in range(7):
  ax.text(j,i,(f'{a[i,j]:.1e}' if i == j else f'{a[i,j]:.4f}'),ha='center',va='center',fontsize=11,color='white' if a[i,j]<a.max()*.5 else 'black')
fig.suptitle('Joint feature distribution distance · multiscale MMD²',fontsize=16,y=.98)
ax.set_title('Seven graphs · 768 raw dimensions · nonzero feature rows',fontsize=11,pad=14)
fig.colorbar(im,ax=ax,label='Unbiased MMD² (lower = more similar)',shrink=.85)
fig.text(.5,.025,'2,000 nodes/graph × 3 repeats · three shared RBF bandwidths\nDiagonal: disjoint same-graph samples (noise baseline); repaired Suspended',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.09,1,.95])
for ext in ['png','pdf']:fig.savefig(ROOT/f'figures/feature_joint_mmd_seven.{ext}',dpi=180)
