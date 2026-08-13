#!/usr/bin/env python3
"""Error audit of the best-donor predictor over every ladder cell.

The composition rule from plot_donor_receiver_vs_ladder.py, applied to EVERY cell:
    pred(mix, T) = max_{D in mix} SPEC[D][T]        (single-source matrix only)
Because the matrix diagonal dominates its column everywhere, this one formula
covers both cases: T in the mix -> its own specialist ceiling; T held out -> its
best in-mix donor. Zero fitted parameters, so in-sample error is honest.

Scored on the unique (mix, target) cells of the 3-order ladder at merge size >= 2:
18 distinct mixes (rung-8 and the A2/B2 mix are shared between orders and counted
once; rung-1 models are specialists, some literally reused matrix rows) x 8
targets = 144 cells, 87 in-mix / 57 held-out.

Panels:
  left   predicted vs observed for all 144 cells, colored by role, with the
         overall and per-role error numbers.
  right  MAE ladder of alternative predictors: the max rule, the max rule with a
         single fitted in-mix dilution tax, the mean-donor rule, a per-target
         constant (no mix information), and "always predict the ceiling".

Reads data/nm_ladder_order_robustness_long.csv and
../nm_single_source_matrix/data/nm_single_source_matrix.csv. Writes figures/.
  /opt/homebrew/bin/python3.11 plot_predictor_error.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "nm_ladder_order_robustness_long.csv")
SS = os.path.join(HERE, "..", "nm_single_source_matrix", "data",
                  "nm_single_source_matrix.csv")
FIGS = os.path.join(HERE, "figures")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
CORAL, CORAL_DK = "#d85a30", "#8a3b1c"   # in-mix cells
GRAY, GRAY_DK = "#8f8d87", "#5f5e5a"     # held-out cells
GREEN = "#2e8b45"

GRAPHS = ["ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
          "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter"]
GLAB = ["ukr", "covid", "midterm", "cov-pol", "elec '20", "ukr-susp", "twibot", "cp-hk"]
LAB = dict(zip(GRAPHS, GLAB))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})


def load():
    rows = []
    with open(DATA, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["auc"] == "":
                continue
            rows.append(dict(order=r["order"], rung=int(r["rung"]), graph=r["test_graph"],
                             auc=float(r["auc"]), entry=int(r["entry_rung"])))
    spec = {}
    with open(SS, newline="") as fh:
        for r in csv.DictReader(fh):
            for g in GRAPHS:
                spec[(r["train_graph"], g)] = float(r[g])
    return rows, spec


def build_cells(rows, spec):
    """Unique (mix, target) cells at merge size >= 2, with predictions."""
    auc = {(r["order"], r["rung"], r["graph"]): r["auc"] for r in rows}
    entry = {(r["order"], r["graph"]): r["entry"] for r in rows}
    cells, seen = [], set()
    for order in ("A", "B", "C"):
        for rung in range(2, 9):
            mix = frozenset(g for g in GRAPHS if entry[(order, g)] <= rung)
            if mix in seen:
                continue
            seen.add(mix)
            for t in GRAPHS:
                cells.append(dict(
                    order=order, rung=rung, t=t, size=len(mix),
                    role="in" if t in mix else "out",
                    obs=auc[(order, rung, t)],
                    p_max=max(spec[(d, t)] for d in mix),
                    p_mean=float(np.mean([spec[(d, t)] for d in mix])),
                    p_ceil=spec[(t, t)],
                ))
    return cells


def errs(obs, pred):
    d = np.asarray(obs) - np.asarray(pred)
    return dict(mae=float(np.mean(np.abs(d))), rmse=float(np.sqrt(np.mean(d ** 2))),
                bias=float(np.mean(d)), p90=float(np.percentile(np.abs(d), 90)),
                mx=float(np.max(np.abs(d))))


def chrome(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows, spec = load()
    cells = build_cells(rows, spec)
    inn = [c for c in cells if c["role"] == "in"]
    out = [c for c in cells if c["role"] == "out"]
    print(f"[cells] {len(cells)} unique (mix, target) cells: "
          f"{len(inn)} in-mix, {len(out)} held-out, "
          f"{len({frozenset() if False else (c['order'], c['rung']) for c in cells})} mixes")

    obs = [c["obs"] for c in cells]
    e_max = errs(obs, [c["p_max"] for c in cells])
    e_in = errs([c["obs"] for c in inn], [c["p_max"] for c in inn])
    e_out = errs([c["obs"] for c in out], [c["p_max"] for c in out])

    # one fitted parameter: the in-mix dilution tax (mean in-mix residual)
    tau = e_in["bias"]
    p_tax = [c["p_max"] + (tau if c["role"] == "in" else 0.0) for c in cells]
    e_tax = errs(obs, p_tax)

    # baselines
    e_meandonor = errs(obs, [c["p_mean"] for c in cells])
    e_ceil = errs(obs, [c["p_ceil"] for c in cells])
    tmean = {t: float(np.mean([c["obs"] for c in cells if c["t"] == t])) for t in GRAPHS}
    e_tmean = errs(obs, [tmean[c["t"]] for c in cells])

    # R^2 vs the grand mean and vs the per-target baseline
    ss = lambda pred: float(np.sum((np.asarray(obs) - np.asarray(pred)) ** 2))
    ss_max = ss([c["p_max"] for c in cells])
    r2_grand = 1 - ss_max / float(np.sum((np.asarray(obs) - np.mean(obs)) ** 2))
    r2_tmean = 1 - ss_max / ss([tmean[c["t"]] for c in cells])

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5), dpi=200,
                             gridspec_kw={"width_ratios": [1.0, 1.05]})
    fig.subplots_adjust(left=0.075, right=0.98, top=0.76, bottom=0.13, wspace=0.30)

    # left: predicted vs observed
    ax = axes[0]
    lims = (0.52, 1.0)
    ax.plot(lims, lims, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=2)
    ax.scatter([c["p_max"] for c in out], [c["obs"] for c in out], s=34, color=GRAY,
               alpha=0.8, edgecolor="white", linewidth=0.5, zorder=5,
               label=f"held-out target (n={len(out)}) — pred = best in-mix donor")
    ax.scatter([c["p_max"] for c in inn], [c["obs"] for c in inn], s=34, color=CORAL,
               alpha=0.8, edgecolor="white", linewidth=0.5, zorder=6,
               label=f"in-mix target (n={len(inn)}) — pred = own ceiling")
    ax.text(0.035, 0.955, f"MAE {e_max['mae']:.3f} · RMSE {e_max['rmse']:.3f} · "
            f"90% of cells within {e_max['p90']:.3f}",
            transform=ax.transAxes, fontsize=9.2, color=INK, fontweight="bold")
    ax.text(0.035, 0.895, f"in-mix: MAE {e_in['mae']:.3f}, bias {e_in['bias']:+.3f} "
            f"(dilution)   ·   held-out: MAE {e_out['mae']:.3f}, bias {e_out['bias']:+.3f}",
            transform=ax.transAxes, fontsize=8.2, color=MUTED)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("predicted AUC = max single-source transfer from the mix",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("observed merged-model AUC", fontsize=9.5, color=INK)
    chrome(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8.2, handlelength=1.0)
    ax.set_title("every unique ladder cell vs the zero-parameter rule",
                 fontsize=10.4, color=INK, loc="left", pad=8)

    # right: MAE ladder of predictors
    ax = axes[1]
    models = [
        (f"max rule + in-mix tax ({tau:+.3f})  [1 fitted param]", e_tax, GREEN),
        ("best-donor (max) rule  [0 params]", e_max, GREEN),
        ("per-target constant (no mix info)", e_tmean, GRAY),
        ("always predict own ceiling", e_ceil, GRAY),
        ("mean-donor rule", e_meandonor, GRAY),
    ]
    y = np.arange(len(models))[::-1]
    for yi, (name, e, col) in zip(y, models):
        ax.barh(yi, e["mae"], height=0.56, color=col, edgecolor="white", zorder=3)
        ax.annotate(f"{e['mae']:.3f}", xy=(e["mae"], yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.6,
                    color=INK, fontweight="bold")
        ax.annotate(name, xy=(0.0015, yi), xytext=(0, 13), textcoords="offset points",
                    va="bottom", ha="left", fontsize=8.6, color=INK)
    ax.set_yticks([])
    ax.set_xlim(0, max(e["mae"] for _, e, _ in models) * 1.18)
    ax.set_xlabel("mean absolute error on the 144 ladder cells  (AUC pts)",
                  fontsize=9.5, color=INK)
    chrome(ax)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_title("the max rule vs alternatives", fontsize=10.4, color=INK,
                 loc="left", pad=8)

    fig.text(0.075, 0.965, "Error of the best-donor predictor over the whole ladder",
             fontsize=13, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.075, 0.915,
             "pred(mix, target) = max over in-mix sources of the single-source "
             "transfer — one formula for both roles (the matrix diagonal dominates "
             "every column) · zero fitted parameters\nscored on the 18 distinct "
             "mixes × 8 targets at merge size ≥ 2 (shared rungs counted once; "
             "rung-1 models are the specialists themselves) · 1 seed",
             fontsize=8.4, color=MUTED, ha="left", va="top", linespacing=1.5)

    for ext in ("pdf", "png"):
        p = os.path.join(FIGS, f"order_predictor_error.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print("wrote", p)
    plt.close(fig)

    # ---------------------------------------------------------------- console
    def line(name, e):
        print(f"  {name:<38} MAE {e['mae']:.4f}  RMSE {e['rmse']:.4f}  "
              f"bias {e['bias']:+.4f}  p90|e| {e['p90']:.4f}  max|e| {e['mx']:.4f}")

    print()
    line("max rule (all cells)", e_max)
    line("  in-mix cells only", e_in)
    line("  held-out cells only", e_out)
    line(f"max rule + in-mix tax {tau:+.4f}", e_tax)
    line("mean-donor rule", e_meandonor)
    line("per-target constant", e_tmean)
    line("always own ceiling", e_ceil)
    print(f"\n  R² of max rule: {r2_grand:.4f} vs grand mean · {r2_tmean:.4f} vs "
          f"per-target constant")

    resid = sorted(cells, key=lambda c: -abs(c["obs"] - c["p_max"]))
    print("\n  worst 6 cells (obs − pred):")
    for c in resid[:6]:
        print(f"    {LAB[c['t']]:<9} {c['role']:<4} order {c['order']} size {c['size']}  "
              f"pred {c['p_max']:.4f}  obs {c['obs']:.4f}  Δ {c['obs']-c['p_max']:+.4f}")


if __name__ == "__main__":
    main()
