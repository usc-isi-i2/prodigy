#!/usr/bin/env python3
"""Figures from aggregate NM bio-cluster/weighting evidence only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = {
    "ukraine": ["Zero bio vector", "C0 Short / mixed multilingual", "C1 Tech / crypto / commerce",
        "C2 Personal / political", "C3 Journalism / media", "C4 Creative / interests",
        "C5 Ukraine / democracy", "C6 Spanish / Romance languages", "C7 Policy / professional"],
    "hongkong": ["Zero bio vector", "C0 Japanese-language / interests", "C1 Hongkonger self-description",
        "C2 Short / mixed multilingual", "C3 Professional / personal", "C4 Rights / anti-CCP vocabulary",
        "C5 News / URL-heavy", "C6 Stand with Hong Kong", "C7 General / values"],
}
COLORS = ["#1673a6", "#c85321"]


def main():
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    data={t:json.loads((ROOT/f"data/nm_bio_clusters_{t}.json").read_text()) for t in LABELS}
    weight={t:json.loads((ROOT/f"data/nm_query_weighting_{t}.json").read_text()) for t in LABELS}
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    for ax,(target,d) in zip(axes,data.items()):
        cs=d["runs"]["8"]["clusters"]
        y=np.arange(len(cs))
        for j,m in enumerate(["ukr","hk"]):
            values=np.array([c["streams"]["test"][m+"_accuracy"]*100 for c in cs])
            ax.scatter(values,y+(j-.5)*.2,c=COLORS[j],marker=["o","s"][j],label=["Ukraine-trained","Hong Kong-trained"][j],zorder=3)
            for v,yy in zip(values,y+(j-.5)*.2):
                ax.annotate(f"{v:.1f}",(v,yy),xytext=(5,0),textcoords="offset points",va="center",fontsize=8)
        ax.set_yticks(y,LABELS[target]);ax.invert_yaxis();ax.set_xlim(0,72)
        ax.set_xlabel("Test query accuracy (%)");ax.set_title(target.capitalize()+" target")
        ax.grid(axis="x",alpha=.2)
    axes[0].legend(loc="lower left",bbox_to_anchor=(0,1.07),frameon=False,ncol=2)
    fig.suptitle("NM bio clusters: model differences depend on the query group",fontsize=15)
    fig.tight_layout(rect=[0,.03,1,.94]);fig.text(.02,.012,"Fixed validation-fitted clusters; occurrence-weighted test scores. Semantic labels are descriptive and boundaries overlap.")
    fig.savefig(ROOT/"figures/nm_bio_cluster_accuracy.png",dpi=180,bbox_inches="tight");plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,target in enumerate(LABELS):
        b=data[target]["baseline"]["test"];w=weight[target]["splits"]["test"]
        metrics=[(b["ukr_accuracy"],b["hk_accuracy"]),
            (w["node_weighted_ukr_accuracy"],w["node_weighted_hk_accuracy"]),
            (w["novel_vs_val"]["ukr_accuracy"],w["novel_vs_val"]["hk_accuracy"])]
        ax=axes[0,j];y=np.arange(3)
        for m in range(2):
            vals=np.array([a[m] for a in metrics])*100
            ax.barh(y+(m-.5)*.3,vals,height=.3,color=COLORS[m],label=["Ukraine-trained","Hong Kong-trained"][m])
            for yy,v in zip(y+(m-.5)*.3,vals):ax.text(v+.7,yy,f"{v:.1f}",va="center",fontsize=9)
        ax.set_yticks(y,["Query occurrences","Equal weight per query node","Queries absent from validation"])
        ax.invert_yaxis();ax.set_xlim(0,65);ax.set_xlabel("Test accuracy (%)");ax.set_title(target.capitalize()+" target")
        ax=axes[1,j];bins=["1","2-4","5-19","20+"]
        for m,key in enumerate(["ukr_accuracy","hk_accuracy"]):
            ax.plot(bins,[w["frequency_groups"][b][key]*100 for b in bins],marker=["o","s"][m],color=COLORS[m])
        ax.set_xlabel("Test occurrences per query node")
        ax.set_ylabel("Accuracy within frequency bin (%)");ax.set_ylim(0,65);ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,ncol=2,loc="lower left",bbox_to_anchor=(0,1.08))
    fig.suptitle("Repeated query nodes change the apparent transfer result",fontsize=15)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(ROOT/"figures/nm_query_weighting.png",dpi=180,bbox_inches="tight");plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharex=True)
    entries=[("both_wrong","support_margin_ukr","Both wrong: Ukraine choice"),
        ("both_wrong","support_margin_hk","Both wrong: Hong Kong choice"),
        ("ukr_only","support_margin_hk","UKR only right: HK choice"),
        ("hk_only","support_margin_ukr","HK only right: UKR choice")]
    for ax,(target,d) in zip(axes,data.items()):
        boxes=[]
        for cohort,key,label in entries:
            v=d["geometry"]["test"][cohort][key];q=v["quantiles"]
            boxes.append(dict(label=label,med=q["0.5"],q1=q["0.25"],q3=q["0.75"],whislo=q["0.1"],whishi=q["0.9"]))
        ax.bxp(boxes,vert=False,showfliers=False)
        ax.axvline(0,color=".6",lw=1);ax.invert_yaxis();ax.set_title(target.capitalize()+" target")
        ax.set_xlabel("Query-to-support GTE cosine margin\n(true class minus predicted class)")
    fig.suptitle("Wrong and true NM classes often have similar bio semantics",fontsize=14)
    fig.tight_layout(rect=[0,.06,1,.93]);fig.text(.05,.02,"Median, interquartile range, and 10–90% interval; exclude comparisons involving zero vectors. Not model logits.")
    fig.savefig(ROOT/"figures/nm_support_cosine_margins.png",dpi=180,bbox_inches="tight");plt.close(fig)


if __name__=="__main__":main()
