"""Regenerate the seed-0 pilot tables and figures from checked-in measurements."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
d = json.loads((DATA / "aggregate.json").read_text())
history, refs = d["history"], d["references"]
sources = ["ukr_rus_twitter", "facebook_page_reference"]
names = {sources[0]: "Ukraine", sources[1]: "Facebook"}
colors = {"extended_bce": "#68727c", "async_kd": "#087c80", "singleton": "#c54b52"}
labels = {"extended_bce": "BCE (rewind replay identical)", "async_kd": "Loss switching", "singleton": "Own-source singleton"}

# A deterministic replay is a control check, not an independent treatment/seed.
extended = {r["logical_step"]: r for r in history["extended_bce"]}
for r in history["rewind_bce"]:
    for metric in ("validation", "training_probe", "counts"):
        assert r[metric] == extended[r["logical_step"]][metric]
assert d["manifest"]["replay_check"]["max_parameter_difference"] == 0
assert all(r["summary"]["historical_model_identity_verified"] for r in refs.values())

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), layout="constrained")
for i, source in enumerate(sources):
    for arm in ["extended_bce", "async_kd"]:
        rows = [r for r in history[arm] if r["logical_step"] > 0]
        auc = [r["validation"][i]["auc"] * 100 for r in rows]
        style = dict(color=colors[arm], lw=1.8, label=labels[arm], alpha=.9)
        axes[0, i].plot([r["counts"][i]["positive_examples"] / 1e6 for r in rows], auc, **style)
        axes[1, i].plot([r["training_probe"][i]["bce"] for r in rows], auc, **style)
    rows = [r for r in refs[source]["history"] if r["step"] > 0]
    auc = [r["validation"]["auc"] * 100 for r in rows]
    style = dict(color=colors["singleton"], ls="--", lw=1.8, label=labels["singleton"])
    axes[0, i].plot([r["counts"]["positive_examples"] / 1e6 for r in rows], auc, **style)
    axes[1, i].plot([r["training_probe"]["bce"] for r in rows], auc, **style)
    selected = next(r for r in rows if r["step"] == refs[source]["summary"]["historical_best_step"])
    kd = d["summary"]["async_kd"]
    end_auc = kd["endpoint"]["validation"][i]["auc"] * 100
    for ax, x in [(axes[0, i], kd["surviving_counts"][i]["positive_examples"] / 1e6),
                  (axes[1, i], kd["endpoint"]["training_probe"][i]["bce"])]:
        ax.scatter([x], [end_auc], s=135, marker="*", color=colors["async_kd"], edgecolor="white", linewidth=.5, zorder=8)
    for ax, x in [(axes[0, i], selected["counts"]["positive_examples"] / 1e6),
                  (axes[1, i], selected["training_probe"]["bce"])]:
        ax.scatter([x], [selected["validation"]["auc"] * 100], s=40, marker="D", color=colors["singleton"], zorder=8)
    for ax in axes[:, i]:
        ax.set_title(names[source])
        ax.set_ylabel("Source validation AUC (%)")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, i].set_xlabel("Source positive examples on retained path (millions)")
    axes[1, i].set_xlabel("Fixed-probe hard-label training BCE (lower →)")
    axes[1, i].set_xscale("log")
    axes[1, i].invert_xaxis()
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, frameon=False)
fig.suptitle("Loss switching: source exposure and training-loss diagnostics\n"
             "Initialization omitted; all later validations shown in time order, including discarded tails.\n"
             "Stars: KD endpoint. Diamonds: original selected singletons. Rewinds can move curves left.", fontsize=12)
fig.savefig(FIGURES / "source_curves.png", dpi=180)
fig.savefig(FIGURES / "source_curves.pdf")
plt.close(fig)

matrix = pd.read_csv(DATA / "matrix.csv")
singletons = pd.read_csv(DATA / "singleton_baselines.csv")
transfer = matrix.pivot(index="target", columns="arm", values="roc_auc") * 100
transfer["Ukraine singleton"] = singletons[singletons.source == sources[0]].set_index("target").auc * 100
transfer["Facebook singleton"] = singletons[singletons.source == sources[1]].set_index("target").auc * 100
transfer["KD minus matched BCE"] = transfer.async_kd - transfer.extended_bce_matched
transfer["KD minus Ukraine singleton"] = transfer.async_kd - transfer["Ukraine singleton"]
transfer["role"] = ["training graph test" if t in sources else "transfer graph test" for t in transfer.index]
transfer.to_csv(DATA / "transfer_comparison.csv")
unseen = transfer[transfer.role == "transfer graph test"]
means = unseen.drop(columns="role").mean().rename("mean_auc_or_delta_pp")
means.to_csv(DATA / "transfer_means.csv")
assert len(unseen) == 6
assert (unseen["KD minus matched BCE"] > 0).all()
assert (unseen.async_kd > unseen.extended_bce).all()

pretty = {"covid19_twitter": "COVID", "covid_political": "COVID political", "cp_hk_twitter": "Hong Kong", "midterm": "Midterm", "twibot20": "TwiBot-20", "ukr_rus_suspended": "Ukraine suspended"}
fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
y = np.arange(len(unseen))
ax.barh(y - .19, unseen["KD minus matched BCE"], height=.35, color=colors["async_kd"], label="vs BCE at same retained input exposure")
ax.barh(y + .19, unseen["KD minus Ukraine singleton"], height=.35, color=colors["singleton"], label="vs stronger constituent singleton (Ukraine)")
ax.set_yticks(y, [pretty[t] for t in unseen.index])
ax.invert_yaxis()
ax.axvline(0, color="#333333", lw=.8)
ax.set_xlabel("AUC difference (percentage points)")
ax.set_title("Loss switching reduces negative transfer, with residual singleton deficits\nSix graphs excluded from joint training; one seed, exploratory evaluation", fontsize=12)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", alpha=.2)
ax.set_axisbelow(True)
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.savefig(FIGURES / "transfer_deltas.png", dpi=180)
fig.savefig(FIGURES / "transfer_deltas.pdf")
plt.close(fig)

endpoints = []
matches = []
for i, source in enumerate(sources):
    kd = d["summary"]["async_kd"]["endpoint"]
    for arm, summary in d["summary"].items():
        endpoints.append(dict(arm=arm, source=source, logical_steps=summary["logical_step"], physical_steps=summary["physical_steps"],
                              validation_auc=summary["endpoint"]["validation"][i]["auc"] * 100,
                              validation_bce=summary["endpoint"]["validation"][i]["bce"],
                              training_bce=summary["endpoint"]["training_probe"][i]["bce"], **summary["surviving_counts"][i]))
    for arm in ("extended_bce", "singleton"):
        candidates = ([dict(step=r["logical_step"], loss=r["training_probe"][i]["bce"], auc=r["validation"][i]["auc"] * 100) for r in history[arm]]
                      if arm != "singleton" else
                      [dict(step=r["step"], loss=r["training_probe"]["bce"], auc=r["validation"]["auc"] * 100) for r in refs[source]["history"]])
        loss = kd["training_probe"][i]["bce"]
        nearest = min(candidates, key=lambda r: abs(r["loss"] - loss))
        matches.append(dict(source=source, comparator=arm, kd_training_bce=loss, kd_validation_auc=kd["validation"][i]["auc"] * 100,
                            nearest_step=nearest["step"], nearest_training_bce=nearest["loss"], nearest_validation_auc=nearest["auc"],
                            absolute_bce_mismatch=abs(nearest["loss"] - loss),
                            kd_minus_nearest_auc_pp=kd["validation"][i]["auc"] * 100 - nearest["auc"]))
pd.DataFrame(endpoints).to_csv(DATA / "source_endpoints.csv", index=False)
pd.DataFrame(matches).to_csv(DATA / "nearest_training_loss.csv", index=False)
print(means.to_string())
print(pd.DataFrame(matches).to_string(index=False))
