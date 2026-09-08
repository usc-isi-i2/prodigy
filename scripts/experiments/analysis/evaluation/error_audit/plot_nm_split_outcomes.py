#!/usr/bin/env python3
"""Paired held-out NM outcome proportions from public aggregates."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--figure-dir",type=Path,required=True)
    a=p.parse_args()
    a.figure_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    groups=[("both_correct","Both correct","#459681"),("ukr_only","Ukraine only correct","#1673a6"),
            ("hk_only","Hong Kong only correct","#c85321"),("both_wrong","Both wrong","#bec4cd")]
    for ax,target in zip(axes,["ukraine","hongkong"]):
        d=json.loads((a.data_dir/f"nm_bio_clusters_{target}.json").read_text())
        left=np.zeros(2)
        for key,label,color in groups:
            values=np.array([d["baseline"][s][key]*100 for s in ["val","test"]])
            ax.barh([0,1],values,left=left,color=color,label=label,height=.52)
            for y,v,l in zip([0,1],values,left):
                if v>3: ax.text(l+v/2,y,f"{v:.1f}",ha="center",va="center",fontsize=9)
            left+=values
        ax.set_yticks([0,1],["Validation","Test"])
        ax.invert_yaxis();ax.set_xlim(0,100)
        ax.set_xlabel("Query occurrences (%)")
        ax.set_title("Ukraine" if target=="ukraine" else "Hong Kong")
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Held-out NM: paired successes and shared failures",fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False)
    fig.tight_layout(rect=[0,.09,1,.94])
    fig.savefig(a.figure_dir/"nm_paired_outcomes.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":main()
