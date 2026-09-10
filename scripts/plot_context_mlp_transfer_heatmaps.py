#!/usr/bin/env python3
"""Render context-MLP transfer heatmaps and a three-view comparison panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from plot_node_only_transfer_heatmaps import LP_TARGETS, SHORT, SOURCES, load_matrix


VIEWS = {
    "node": "Node only",
    "neighborhood": "Neighborhood only",
    "node_neighborhood": "Node + neighborhood",
}


def axes_style(ax, targets, show_ylabel=True):
    ax.set_xticks(range(len(targets)), [SHORT[x] for x in targets], rotation=38, ha="right")
    ax.set_yticks(range(len(SOURCES)), [SHORT[x] for x in SOURCES] if show_ylabel else [])
    ax.set_xlabel("Evaluation target")
    if show_ylabel:
        ax.set_ylabel("Training source")
    ax.tick_params(length=0, labelsize=9)
    for row, source in enumerate(SOURCES):
        if source in targets:
            col = targets.index(source)
            ax.add_patch(Rectangle((col-.49,row-.49),.98,.98,fill=False,edgecolor="white",linewidth=2))


def annotations(ax, values, fmt):
    midpoint=(float(np.nanmin(values))+float(np.nanmax(values)))/2
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color="white" if values[row,col] > midpoint else "black"
            ax.text(col,row,format(values[row,col],fmt),ha="center",va="center",fontsize=8,color=color)


def single(matrix, view, objective, output):
    is_fp=objective=="fp"; values=matrix.to_numpy()*(100 if is_fp else 1)
    figsize=(9.3,7.2) if is_fp else (7.8,7.2)
    fig,ax=plt.subplots(figsize=figsize)
    if is_fp:
        image=ax.imshow(values,cmap="magma_r",aspect="auto",vmin=0,vmax=33)
        subtitle="Scaled cosine error ×100 · lower is better"
        label="Scaled cosine error ×100"; fmt=".1f"
    else:
        image=ax.imshow(values,cmap="viridis",aspect="auto",vmin=.49,vmax=.89)
        subtitle="ROC-AUC · higher is better"
        label="ROC-AUC"; fmt=".3f"
    annotations(ax,values,fmt); axes_style(ax,list(matrix.columns))
    ax.set_title(f"{VIEWS[view]} — {'feature prediction' if is_fp else 'static link prediction'}",loc="left",weight="bold",pad=30)
    ax.text(0,1.015,subtitle+" · White boxes mark source=target",transform=ax.transAxes,fontsize=9,color=".35")
    bar=fig.colorbar(image,ax=ax,shrink=.8,pad=.025); bar.set_label(label)
    fig.tight_layout(); output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=240,bbox_inches="tight"); plt.close(fig)


def overview(matrices, output):
    fig,axes=plt.subplots(2,3,figsize=(18,11),gridspec_kw={"width_ratios":[1.35,1.35,1.35]})
    for col,view in enumerate(VIEWS):
        for row,objective in enumerate(("fp","lp")):
            ax=axes[row,col]; matrix=matrices[(view,objective)]
            values=matrix.to_numpy()*(100 if objective=="fp" else 1)
            image=ax.imshow(values,cmap="magma_r" if objective=="fp" else "viridis",aspect="auto",vmin=0 if objective=="fp" else .49,vmax=33 if objective=="fp" else .89)
            axes_style(ax,list(matrix.columns),show_ylabel=col==0)
            ax.set_title(VIEWS[view],weight="bold",fontsize=13)
            if col==2:
                color_axis=fig.add_axes([.945,.595 if objective=="fp" else .145,.012,.245])
                bar=fig.colorbar(image,cax=color_axis)
                bar.set_label("Error ×100 ↓" if objective=="fp" else "ROC-AUC ↑")
    fig.text(.01,.72,"Feature prediction",rotation=90,va="center",weight="bold",fontsize=14)
    fig.text(.01,.27,"Static link prediction",rotation=90,va="center",weight="bold",fontsize=14)
    fig.suptitle("MLP transfer matrices by available input view",x=.04,ha="left",weight="bold",fontsize=18)
    fig.text(.04,.945,"Consistent color scales across views · White boxes mark source=target",color=".35",fontsize=10)
    fig.subplots_adjust(left=.07,right=.92,bottom=.09,top=.9,wspace=.24,hspace=.32)
    output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,dpi=220,bbox_inches="tight"); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--node-root",type=Path,required=True); p.add_argument("--context-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    paths={
        ("node","fp"):a.node_root/"node_mlp_fp_transfer.tsv", ("node","lp"):a.node_root/"node_mlp_lp_transfer.tsv",
        ("neighborhood","fp"):a.context_root/"neighborhood_fp_transfer.tsv", ("neighborhood","lp"):a.context_root/"neighborhood_lp_transfer.tsv",
        ("node_neighborhood","fp"):a.context_root/"node_neighborhood_fp_transfer.tsv", ("node_neighborhood","lp"):a.context_root/"node_neighborhood_lp_transfer.tsv",
    }
    matrices={(view,obj):load_matrix(path,SOURCES if obj=="fp" else LP_TARGETS) for (view,obj),path in paths.items()}
    for view in ("neighborhood","node_neighborhood"):
        for obj in ("fp","lp"):
            single(matrices[(view,obj)],view,obj,a.output_dir/f"{view}_mlp_{obj}_transfer_heatmap.png")
    overview(matrices,a.output_dir/"mlp_three_view_transfer_overview.png")

if __name__=="__main__": main()
