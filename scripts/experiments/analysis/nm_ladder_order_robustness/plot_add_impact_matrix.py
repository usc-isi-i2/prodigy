#!/usr/bin/env python3
"""Mean impact of adding each source, broken down by target graph (8x8 matrices).

Every source is added exactly once per order; the addition is measurable when it
happens at rung >= 2 (a rung-1 entry has no "before"), giving n=3 events for
midterm / cov-pol / ukr-susp / twibot / cp-hk and n=2 for ukr / covid / elec'20
(21 events total). The impact of one event on one target is
    Delta AUC(target) = AUC(rung with the source) - AUC(rung just below).

Three panels over the same 21 events:
  pooled       cell = mean Delta over all measurable additions of that source.
               Diagonal = the source's own entry (newcomer) boost.
  target in    only events where the target was ALREADY in the merge -> the
               per-target version of the coral "in-mix" bars (interference).
  target out   only events where the target was NOT YET added -> the per-target
               version of the gray "out-of-mix" bars (spillover). Newcomer
               (diagonal) events are excluded from both split panels.

Caveat the figure states: the big out-of-mix positives all come from order C's
early steps, where every not-yet-added graph recovers as coverage grows
(headroom recovery, see order_heldout_headroom) -- they are not donor-specific
transfer. Color scale is clipped so off-diagonal structure stays visible; the
diagonal saturates and its value is annotated.

Second figure (order_add_impact_by_source): the same events aggregated over
targets -- per source added, three bars (own entry boost / mean on in-mix
targets / mean on out-of-mix targets), whisker = min/max over that source's
addition events. The row-sum view of the matrix, in the regret-bar style.

Reads data/nm_ladder_order_robustness_long.csv; writes figures/. Local python:
  /opt/homebrew/bin/python3.11 plot_add_impact_matrix.py
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "nm_ladder_order_robustness_long.csv")
FIGS = os.path.join(HERE, "figures")

GREEN, CORAL = "#2e8b45", "#d85a30"      # + toward best / - away (regret-bar palette)
GREEN_DK, CORAL_DK = "#1a5a2a", "#8a3b1c"
GRAY, GRAY_DK = "#8f8d87", "#5f5e5a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
EMPTY = "#f1f0ea"
ORDERS = ("A", "B", "C")

# canonical column order, as everywhere else in the ladder analyses
GRAPHS = ["ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
          "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter"]
GLAB = ["ukr", "covid", "midterm", "cov-pol", "elec '20", "ukr-susp", "twibot", "cp-hk"]
ADD2GRAPH = {"ukr_rus": "ukr_rus_twitter", "covid": "covid19_twitter",
             "midterm": "midterm", "covid_political": "covid_political",
             "election2020": "election2020", "ukr_rus_suspended": "ukr_rus_suspended",
             "twibot20": "twibot20", "cp_hk": "cp_hk_twitter"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})


def load_events():
    """21 measurable addition events: (source graph added, per-target Delta, role)."""
    rows = []
    with open(DATA, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["auc"] == "":
                continue
            rows.append(dict(order=r["order"], rung=int(r["rung"]), graph=r["test_graph"],
                             auc=float(r["auc"]), entry=int(r["entry_rung"]),
                             added=r["added"]))
    auc = {(r["order"], r["rung"], r["graph"]): r["auc"] for r in rows}
    entry = {(r["order"], r["graph"]): r["entry"] for r in rows}
    added_at = {(r["order"], r["rung"]): ADD2GRAPH[r["added"]] for r in rows}

    events = []
    for order in ORDERS:
        for r in range(2, 9):
            src = added_at[(order, r)]
            deltas, roles = {}, {}
            for g in GRAPHS:
                deltas[g] = auc[(order, r, g)] - auc[(order, r - 1, g)]
                e = entry[(order, g)]
                roles[g] = "newcomer" if e == r else ("in" if e < r else "out")
            events.append(dict(order=order, rung=r, src=src, deltas=deltas, roles=roles))
    return events


def matrices(events):
    """pooled / in-mix-only / out-only mean matrices + per-cell counts (numpy, nan=empty)."""
    acc = {k: defaultdict(list) for k in ("all", "in", "out")}
    for ev in events:
        for g in GRAPHS:
            acc["all"][(ev["src"], g)].append(ev["deltas"][g])
            if ev["roles"][g] in ("in", "out"):
                acc[ev["roles"][g]][(ev["src"], g)].append(ev["deltas"][g])

    def mat(key):
        m = np.full((8, 8), np.nan)
        n = np.zeros((8, 8), dtype=int)
        for i, s in enumerate(GRAPHS):
            for j, g in enumerate(GRAPHS):
                vals = acc[key][(s, g)]
                if vals:
                    m[i, j] = float(np.mean(vals))
                    n[i, j] = len(vals)
        return m, n

    return {k: mat(k) for k in ("all", "in", "out")}


def draw_panel(ax, m, n, title, vmax, show_n, ylabels):
    cmap = LinearSegmentedColormap.from_list("impact", [CORAL, "#ffffff", GREEN])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    body = np.ma.masked_invalid(np.clip(m, -vmax, vmax))
    cmap.set_bad(EMPTY)
    im = ax.imshow(body, cmap=cmap, norm=norm, aspect="equal")

    for i in range(8):
        for j in range(8):
            if np.isnan(m[i, j]):
                ax.text(j, i, "·", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            sat = abs(m[i, j]) > 0.62 * vmax
            ax.text(j, i, f"{m[i, j]:+.3f}".replace("+0.", "+.").replace("-0.", "−."),
                    ha="center", va="center", fontsize=7.4,
                    color="white" if sat else INK,
                    fontweight="bold" if i == j or abs(m[i, j]) >= 0.01 else "normal")
            if show_n:
                ax.text(j + 0.42, i + 0.40, str(n[i, j]), ha="right", va="bottom",
                        fontsize=5.4, color="white" if sat else MUTED)

    for i in range(8):        # outline the diagonal = the newcomer's own entry boost
        if not np.isnan(m[i, i]):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=INK, lw=1.4, zorder=5))

    ax.set_xticks(range(8))
    ax.set_xticklabels(GLAB, fontsize=8, rotation=45, ha="right", color=MUTED)
    ax.set_yticks(range(8))
    ax.set_yticklabels(ylabels if ylabels else [], fontsize=8, color=MUTED)
    if not ylabels:
        ax.tick_params(left=False)
    ax.set_xticks(np.arange(-0.5, 8), minor=True)
    ax.set_yticks(np.arange(-0.5, 8), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")
    return im


def fig_by_source(events):
    """Aggregate over targets: per source added, own boost / in-mix / out-of-mix bars."""
    per = {g: {"new": [], "in": [], "out": []} for g in GRAPHS}
    for ev in events:
        per[ev["src"]]["new"].append(ev["deltas"][ev["src"]])
        for role in ("in", "out"):
            vals = [ev["deltas"][g] for g in GRAPHS if ev["roles"][g] == role]
            if vals:
                per[ev["src"]][role].append(float(np.mean(vals)))

    def stats(role):
        vs = [per[g][role] for g in GRAPHS]
        mu = [float(np.mean(v)) for v in vs]
        err = [[m - min(v) for m, v in zip(mu, vs)], [max(v) - m for m, v in zip(mu, vs)]]
        return mu, err, [len(v) for v in vs]

    n_mu, n_err, n_n = stats("new")
    i_mu, i_err, i_n = stats("in")
    o_mu, o_err, o_n = stats("out")

    x = np.arange(8)
    w = 0.27
    ylim = (min(i_mu + o_mu) - 0.022,
            max(m + e for m, e in zip(n_mu, n_err[1])) * 1.14)

    fig, ax = plt.subplots(figsize=(10.6, 5.7), dpi=200, constrained_layout=True)
    ax.axhline(0.0, color="#9c9a93", lw=1.0, zorder=2)
    ax.bar(x - w, n_mu, width=w, color=GREEN, edgecolor="white", linewidth=0.7,
           zorder=3, label="the source's own graph — its entry boost")
    ax.bar(x, i_mu, width=w, color=CORAL, edgecolor="white", linewidth=0.7,
           zorder=3, label="in-mix targets — mean Δ (already in the merge)")
    ax.bar(x + w, o_mu, width=w, color=GRAY, edgecolor="white", linewidth=0.7,
           zorder=3, label="out-of-mix targets — mean Δ (not yet added)")

    for xs, mu, err, n, col in ((x - w, n_mu, n_err, n_n, GREEN_DK),
                                (x, i_mu, i_err, i_n, CORAL_DK),
                                (x + w, o_mu, o_err, o_n, GRAY_DK)):
        keep = [k for k in range(8) if n[k] > 1]
        ax.errorbar([xs[k] for k in keep], [mu[k] for k in keep],
                    yerr=[[err[0][k] for k in keep], [err[1][k] for k in keep]],
                    fmt="none", ecolor=col, elinewidth=1.0, capsize=2.4, zorder=5)

    for xi, m, e in zip(x - w, n_mu, n_err[1]):
        ax.annotate(f"+{m:.3f}", xy=(xi, m + e), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.4, color=GREEN_DK, fontweight="bold")
    for xi, n in zip(x + w, o_n):        # events that still had held-out targets
        ax.annotate(f"n={n}", xy=(xi, ylim[0]), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.4, color=MUTED)

    ax.set_ylim(*ylim)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_ylabel("mean Δ NM AUC at the addition step   (+ = toward best)",
                  fontsize=10, color=INK)
    ax.set_xlim(-0.6, 7.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{l}" for l in GLAB], fontsize=9.5)
    ax.set_xlabel("source added to the SSL pre-training merge", fontsize=10.5, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9, borderaxespad=0.4,
              handlelength=1.2, labelspacing=0.45)
    ax.set_title("Adding a source mostly helps itself — averaged over its addition events",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=24)
    ax.text(0.0, 1.015, "NM 3-shot / 30-way · matched step 40k · bar = mean over the "
            "source's measurable additions (3 orders; 2 for ukr / covid / elec '20), "
            "whisker = min/max over those events\ngreen = its own graph's jump · coral / "
            "gray = mean Δ over the other graphs by role · n = events with any held-out "
            "target left · gray upside = order-C headroom recovery",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.4, color=MUTED)

    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"order_add_impact_by_source.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)
    plt.close(fig)

    print("\naggregated over targets  (mean over events [min..max], n)")
    print("  source      own boost                in-mix targets           out-of-mix targets")
    for k in range(8):
        own = f"{n_mu[k]:+.4f} [{n_mu[k]-n_err[0][k]:+.3f}..{n_mu[k]+n_err[1][k]:+.3f}] n={n_n[k]}"
        im = f"{i_mu[k]:+.4f} [{i_mu[k]-i_err[0][k]:+.3f}..{i_mu[k]+i_err[1][k]:+.3f}] n={i_n[k]}"
        om = f"{o_mu[k]:+.4f} [{o_mu[k]-o_err[0][k]:+.3f}..{o_mu[k]+o_err[1][k]:+.3f}] n={o_n[k]}"
        print(f"  +{GLAB[k]:<10} {own:<24} {im:<24} {om}")


def main():
    os.makedirs(FIGS, exist_ok=True)
    events = load_events()
    mats = matrices(events)
    m_all, n_all = mats["all"]
    m_in, n_in = mats["in"]
    m_out, n_out = mats["out"]

    off = ~np.eye(8, dtype=bool)
    vmax = max(0.04, float(np.nanmax(np.abs(m_all[off]))) * 1.05)

    row_n = [int(n_all[i, i]) for i in range(8)]
    ylabels = [f"+{l}  (n={n})" for l, n in zip(GLAB, row_n)]

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.4), dpi=200)
    fig.subplots_adjust(left=0.095, right=0.895, top=0.775, bottom=0.115, wspace=0.08)
    im = draw_panel(axes[0], m_all, n_all, "all addition events (pooled mean)",
                    vmax, show_n=False, ylabels=ylabels)
    draw_panel(axes[1], m_in, n_in, "target already in the merge (in-mix)",
               vmax, show_n=True, ylabels=None)
    draw_panel(axes[2], m_out, n_out, "target still held out (out-of-mix)",
               vmax, show_n=True, ylabels=None)
    axes[1].set_xlabel("target graph", fontsize=10.5, color=INK, labelpad=8)
    axes[0].set_ylabel("source added to the merge", fontsize=10.5, color=INK)

    cax = fig.add_axes([0.915, 0.18, 0.011, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"mean Δ NM AUC at the addition step\n(color clipped at ±{vmax:.2f}; "
                 "saturated cells keep their printed value)", fontsize=8.4, color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.outline.set_visible(False)

    fig.text(0.06, 0.965, "Mean impact of adding each source, by target graph",
             fontsize=13, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.925,
             "NM 3-shot / 30-way · matched step 40k · cell = mean Δ AUC (rung with the "
             "source − rung just below) over that source's measurable additions (n=3 "
             "orders; n=2 where it opened an order — a rung-1 entry has no 'before') · "
             "boxed diagonal = the newcomer's own entry boost\nsplit panels = the same "
             "events restricted by the target's role at the step (tiny corner number = "
             "n; · = no such event) · the out-of-mix positives are order-C headroom "
             "recovery, not donor-specific transfer (see order_heldout_headroom)",
             fontsize=8.4, color=MUTED, ha="left", va="top", linespacing=1.5)

    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"order_add_impact_matrix.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)
    plt.close(fig)

    # ---------------------------------------------------------------- console
    def show(mat, cnt, head):
        print(f"\n{head}  (rows = source added, cols = target)")
        print("          " + "".join(f"{l:>10}" for l in GLAB))
        for i, l in enumerate(GLAB):
            cells = "".join("      ·   " if np.isnan(mat[i, j]) else f"{mat[i, j]:+9.4f} "
                            for j in range(8))
            print(f"  +{l:<8}{cells}")

    print("\nevents per source: " + ", ".join(
        f"+{GLAB[i]}: " + "/".join(f"{e['order']}r{e['rung']}" for e in events
                                   if e["src"] == GRAPHS[i]) for i in range(8)))
    show(m_all, n_all, "POOLED mean Δ AUC")
    show(m_in, n_in, "IN-MIX-only mean Δ AUC (target already in the merge)")
    show(m_out, n_out, "OUT-only mean Δ AUC (target still held out)")

    fig_by_source(events)


if __name__ == "__main__":
    main()
