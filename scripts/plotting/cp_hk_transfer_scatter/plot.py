"""
Cross-model NM transfer scatter (3-shot, 3-way NM ROC-AUC).

x-axis: evaluation (test) graph
each dot: one trained model evaluated on that test graph

Other models come from experiments_jun_24.tsv (+nm columns), which are 3-WAY
NM ROC-AUC (verified: covid->ukr = 0.986 here vs 0.9245 in the fair 30-way
matrix, so this table is the near-ceiling 3-way eval). CP-HK points are pulled
from the cp_hk_twitter transfer-out experiment at the MATCHING 3-way setting
(NOT the 30-way bar chart, which would be a different scale).
"""
import matplotlib.pyplot as plt

# test graph -> pretty label
tests = [
    ("ukr_rus_twitter", "UKR/RUS"),
    ("cp_hk_twitter",   "CP-HK"),
    ("covid19_twitter", "COVID"),
    ("midterm",         "Midterm"),
    ("covid_political", "Covid pol."),
    ("ukr_rus_suspended","Susp."),
    ("election2020",    "Election"),
]
xlabels = [t[1] for t in tests]
keys = [t[0] for t in tests]

# model -> {test_key: 3-way NM ROC-AUC}
models = {
    "COVID (NM)":            dict(ukr_rus_twitter=.986, covid19_twitter=.998, midterm=.971, covid_political=.738, ukr_rus_suspended=.802, election2020=.757),
    "UKR/RUS (NM)":          dict(ukr_rus_twitter=.992, covid19_twitter=.991, midterm=.956, covid_political=.730, ukr_rus_suspended=.754, election2020=.738),
    "COVID→UKR (seq NM)":    dict(ukr_rus_twitter=.978, covid19_twitter=.952, midterm=.849, covid_political=.684, ukr_rus_suspended=.691, election2020=.671),
    "Merged UCM (NM)":       dict(ukr_rus_twitter=.985, covid19_twitter=.990, midterm=.942, covid_political=.717, ukr_rus_suspended=.780, election2020=.713),
    "Merged UCM (NM+FM)":    dict(ukr_rus_twitter=.961, covid19_twitter=.991, midterm=.924, covid_political=.900, ukr_rus_suspended=.841, election2020=.888),
    "CP-HK (NM)":            dict(ukr_rus_twitter=.8798, cp_hk_twitter=.8627, covid19_twitter=.8226, midterm=.8131, covid_political=.7165, ukr_rus_suspended=.7085, election2020=.6731),
}

# --- HK-in points (test = cp_hk_twitter) from cp_hk_transfer_in, 3-shot 3-way NM ROC-AUC.
# NB: these come from the fair-matrix checkpoints (nm_matrix_*), a later/standardized
# NM run than the jun_24 checkpoints on the other x-ticks — same metric/scale, not the
# identical model. Only single-source lines have a clean counterpart.
models["COVID (NM)"]["cp_hk_twitter"]    = .7544  # nm_matrix_covid
models["UKR/RUS (NM)"]["cp_hk_twitter"]  = .7454  # nm_matrix_ukr
models["Merged UCM (NM)"]["cp_hk_twitter"] = .7376  # nm_matrix_merged_match (ukr+covid proxy)

order = list(models.keys())
cmap = plt.get_cmap("tab10")
colors = {m: cmap(i) for i, m in enumerate(order)}
colors["CP-HK (NM)"] = "#d1495b"  # highlight

x = list(range(len(keys)))
fig, ax = plt.subplots(figsize=(11, 6))

for m in order:
    ys = [models[m].get(k, float("nan")) for k in keys]
    is_cp = m == "CP-HK (NM)"
    ax.plot(x, ys,
            marker=("*" if is_cp else "o"),
            markersize=(20 if is_cp else 11),
            linewidth=(2.4 if is_cp else 1.0),
            linestyle=("-" if is_cp else "--"),
            alpha=(1.0 if is_cp else 0.55),
            color=colors[m], label=m, zorder=(5 if is_cp else 3))

ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=11)
ax.set_ylabel("NM ROC-AUC  (3-shot, 3-way)", fontsize=12)
ax.set_title("Cross-model neighbor-matching transfer by test graph\n"
             "CP-HK added from transfer-out experiment (3-way, matched to table)",
             fontsize=13)
ax.set_ylim(0.60, 1.01)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Train model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = "scripts/plotting/cp_hk_transfer_scatter/cp_hk_cross_model_scatter.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
