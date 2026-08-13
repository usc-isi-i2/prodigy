#!/usr/bin/env python3
"""
Build a tidy per-(source, target) transfer-analysis table for the 8x8 single-source
NM matrix, joined with pairwise graph divergence and per-graph attributes.

Inputs (repo-relative):
  - scripts/experiments/analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/data/nm_single_source_matrix.csv   (8x8 ROC-AUC; rows=train/source, cols=test/target)
  - scripts/experiments/analysis/graph_characterization/statistics/graph_divergence/data/graph_divergence_data.json               (pairwise divergence + per-graph stats)

Output:
  - scripts/experiments/analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/data/transfer_pairs.csv     (64 rows = 8 sources x 8 targets)

Every column corresponds to something in the symmetry / regret / similarity-control
analysis. See README.md in this folder for the column dictionary.
Run from repo root:  python3 scripts/experiments/analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/build_transfer_pairs.py
"""
import csv, json, os, statistics as st

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
AUC_CSV = os.path.join(REPO, "scripts/experiments/analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/data/nm_single_source_matrix.csv")
DIV_JSON = os.path.join(REPO, "scripts/experiments/analysis/graph_characterization/statistics/graph_divergence/data/graph_divergence_data.json")
OUT_CSV = os.path.join(os.path.dirname(__file__), "transfer_pairs.csv")

# ---- transfer matrix (rows=source, cols=target) ----
rows = list(csv.reader(open(AUC_CSV)))
targets = rows[0][1:]
AUC, ceiling = {}, {}
for r in rows[1:]:
    s = r[0]
    for j, t in enumerate(targets):
        AUC[(s, t)] = float(r[j + 1])
    ceiling[s] = AUC[(s, s)]
graphs = [r[0] for r in rows[1:]]            # canonical order = matrix row order

# ---- divergence data ----
dv = json.load(open(DIV_JSON))
DG = dv["graphs"]                            # divergence graph order (may differ)
PW = dv["pairwise"]
METRICS = list(PW.keys())                    # indegree_ks, outdegree_ks, feat_centroid_cosdist, feat_frechet, feat_mmd2, proxy_a_distance
di = {g: k for k, g in enumerate(DG)}
def div(metric, a, b):                       # directed divergence a->b as stored (row=source)
    return PW[metric][di[a]][di[b]]
def div_sym(metric, a, b):
    return 0.5 * (div(metric, a, b) + div(metric, b, a))
feat_hom = {g: dv["per_graph"][g]["feature_homophily"] for g in DG}
label_hom = {g: dv["per_graph"][g].get("label_homophily") for g in DG}

SIM = "proxy_a_distance"                     # primary similarity axis (feature-cloud separability)

# ---- per-graph aggregates (off-diagonal) ----
centrality   = {g: st.mean(div_sym(SIM, g, h) for h in graphs if h != g) for g in graphs}   # mean feat-distance to others (LOW=central)
outflow      = {g: st.mean(AUC[(g, h)] for h in graphs if h != g) for g in graphs}          # source strength (mean AUC out)
inflow       = {g: st.mean(AUC[(h, g)] for h in graphs if h != g) for g in graphs}          # target reachability (mean AUC in)
donor_regret = {g: st.mean(ceiling[h] - AUC[(g, h)] for h in graphs if h != g) for g in graphs}  # mean regret as donor
recv_regret  = {g: st.mean(ceiling[g] - AUC[(h, g)] for h in graphs if h != g) for g in graphs}  # mean regret as target

# ---- within-target donor rank (1 = best foreign source into that target) ----
donor_rank = {}
for t in graphs:
    order = sorted([a for a in graphs if a != t], key=lambda a: -AUC[(a, t)])
    for k, a in enumerate(order):
        donor_rank[(a, t)] = k + 1

def rnd(x, n=6):
    return None if x is None else round(x, n)

COLUMNS = [
    "source", "target", "is_self",
    # --- core transfer / ceilings ---
    "transfer_auc", "source_ceiling", "target_ceiling",
    # --- regret (subtractive + normalized) & retention ---
    "regret", "regret_norm", "retention", "donor_rank",
    # --- directional asymmetry (this dir vs reverse dir) ---
    "reverse_auc", "auc_asym", "regret_reverse", "regret_asym", "retention_reverse", "retention_asym",
    # --- pairwise divergence / similarity ---
    "proxy_a_distance", "proxy_a_distance_sym", "feat_frechet", "feat_mmd2", "feat_centroid_cosdist",
    "indegree_ks", "outdegree_ks", "homophily_gap_signed",
    # --- per-graph attributes broadcast onto the pair ---
    "source_centrality", "target_centrality", "source_outflow", "target_inflow",
    "source_donor_regret_mean", "target_recv_regret_mean",
    "source_feat_homophily", "target_feat_homophily", "source_label_homophily", "target_label_homophily",
]

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    w.writeheader()
    for s in graphs:
        for t in targets:
            self = (s == t)
            auc = AUC[(s, t)]; rev = AUC[(t, s)]
            reg = ceiling[t] - auc
            reg_rev = ceiling[s] - rev
            ret = auc / ceiling[t]
            ret_rev = rev / ceiling[s]
            w.writerow({
                "source": s, "target": t, "is_self": self,
                "transfer_auc": rnd(auc, 4), "source_ceiling": rnd(ceiling[s], 4), "target_ceiling": rnd(ceiling[t], 4),
                "regret": rnd(reg, 4), "regret_norm": rnd(reg / ceiling[t], 4), "retention": rnd(ret, 4),
                "donor_rank": ("" if self else donor_rank[(s, t)]),
                "reverse_auc": rnd(rev, 4), "auc_asym": rnd(auc - rev, 4),
                "regret_reverse": rnd(reg_rev, 4), "regret_asym": rnd(reg - reg_rev, 4),
                "retention_reverse": rnd(ret_rev, 4), "retention_asym": rnd(ret - ret_rev, 4),
                "proxy_a_distance": rnd(div(SIM, s, t)), "proxy_a_distance_sym": rnd(div_sym(SIM, s, t)),
                "feat_frechet": rnd(div_sym("feat_frechet", s, t)), "feat_mmd2": rnd(div_sym("feat_mmd2", s, t)),
                "feat_centroid_cosdist": rnd(div_sym("feat_centroid_cosdist", s, t)),
                "indegree_ks": rnd(div_sym("indegree_ks", s, t)), "outdegree_ks": rnd(div_sym("outdegree_ks", s, t)),
                "homophily_gap_signed": rnd(feat_hom[s] - feat_hom[t]),
                "source_centrality": rnd(centrality[s]), "target_centrality": rnd(centrality[t]),
                "source_outflow": rnd(outflow[s], 4), "target_inflow": rnd(inflow[t], 4),
                "source_donor_regret_mean": rnd(donor_regret[s], 4), "target_recv_regret_mean": rnd(recv_regret[t], 4),
                "source_feat_homophily": rnd(feat_hom[s]), "target_feat_homophily": rnd(feat_hom[t]),
                "source_label_homophily": rnd(label_hom[s]), "target_label_homophily": rnd(label_hom[t]),
            })

print(f"wrote {OUT_CSV}  ({len(graphs)*len(targets)} rows, {len(COLUMNS)} cols)")
