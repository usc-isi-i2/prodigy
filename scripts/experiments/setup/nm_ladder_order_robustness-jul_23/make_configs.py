#!/usr/bin/env python3
"""Generate train configs for the NM ladder order-robustness experiment.

Every rung of every order trains on the SAME all8 merged graph, restricted to that
rung's sources via --neighbor_sampling_source_subset. That is equivalent to training on
the sub-merge (disjoint block-concat => no cross-source edges; episodes are already
confined to one source), so no nested merges are built.

Rungs whose SOURCE SET has already been trained -- by the published order-A ladder, by
the single-source matrix, or by an earlier order in this run -- are reported as reuse
rather than re-emitted. Set identity is what matters, not the path taken to it.

Usage:
    python3 make_configs.py              # write configs + manifest.csv
    python3 make_configs.py --dry-run    # print the plan only
"""
import argparse
import csv
from pathlib import Path

# The all8 merge, in graph_id order (scripts/graph_construction/
# merge_ukr_rus_covid_midterm_all8.yaml). `key` is the name carried in
# graph.source_graph_names, which is what the subset knob resolves.
SOURCES = [
    # key,                dataset_key,          canonical,                single-source run
    ("ukr_rus",           "ukr_rus_twitter",    "ukraine",                "nm_ss_ukr_rus_twitter"),
    ("covid",             "covid19_twitter",    "covid",                  "nm_ss_covid19_twitter"),
    ("midterm",           "midterm",            "midterm",                "nm_ss_midterm"),
    ("covid_political",   "covid_political",    "covid-political",        "nm_ss_covid_political"),
    ("election2020",      "election2020",       "election2020-political", "nm_ss_election2020"),
    ("ukr_rus_suspended", "ukr_rus_suspended",  "ukraine-suspended",      "nm_ss_ukr_rus_suspended"),
    ("twibot20",          "twibot20",           "twibot20",               "nm_ss_twibot20"),
    ("cp_hk",             "cp_hk_twitter",      "hongkong",               "nm_ss_cp_hk_twitter"),
]
KEYS = [s[0] for s in SOURCES]
SS_RUN = {s[0]: s[3] for s in SOURCES}

# Donor strength = mean off-diagonal transfer as a source, from
# analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/data/nm_single_source_matrix.csv. Orders B and C are
# this ranking descending and ascending; A is the published topical order.
ORDERS = {
    "A": ["ukr_rus", "covid", "midterm", "covid_political",
          "election2020", "ukr_rus_suspended", "twibot20", "cp_hk"],
    "B": ["covid", "ukr_rus", "twibot20", "midterm",
          "ukr_rus_suspended", "cp_hk", "covid_political", "election2020"],
}
ORDERS["C"] = list(reversed(ORDERS["B"]))

# Already-trained order-A rungs (the published ladder). Keyed by source set.
EXISTING_LADDER = {
    1: "ukr_only_nm",
    2: "merged_ukr_rus_covid_nm_wb",
    3: "merged_ukr_rus_covid_midterm_nm_wb",
    4: "nm_ladder_4src",
    5: "nm_ladder_5src",
    6: "nm_ladder_6src",
    7: "nm_ladder_7src",
    8: "merged_ukr_rus_covid_midterm_all8_nm_wb",
}

ALL8_ROOT = "/dataMeR1/phil/data/merged/graphs"
ALL8_GRAPH = "ukr_rus_covid_midterm_all8_retweet_graph.pt"

CONFIG_TEMPLATE = """\
# {title}
# Order {order} ({order_desc}), rung {rung}/8 -- adds {added} ({added_canonical}).
# Sources: {sources_pretty}
#
# Trains on the all8 merged graph restricted to this rung's sources via
# neighbor_sampling_source_subset. Equivalent to training on the sub-merge: the merge is
# a disjoint block-concat, so sampled neighborhoods never leave their source graph.
# Otherwise an exact clone of covid_ukr/merged_ukr_rus_covid_midterm_all8_nm.yaml
# (256.S,U,M, no aug, attr_regression_weight=0, within-source balanced episodes), with
# epochs:5 / checkpoint_step:10000 so the run self-terminates with state_dict_40000 as
# its final checkpoint (matched-40k, same as nm_ladder_fillin).
dataset: covid19_twitter
root: {root}
graph_filename: {graph}
task_name: neighbor_matching

edge_view: default
feature_subset: all
original_features: true

n_way: 30
n_shots: 3
n_query: 4
batch_size: 1
dataset_len_cap: 10000
val_len_cap: 500
test_len_cap: 500

# within-balanced episodes, restricted to this rung's sources
neighbor_sampling_episode_source: graph_id
neighbor_sampling_episode_source_weighting: balanced
neighbor_sampling_source_subset: {subset}

epochs: 5                # 5 x 10k = 50k planned; 40k is the final saved ckpt
eval_step: 100000        # > total steps => skip periodic val eval
checkpoint_step: 10000   # ckpts at 10k / 20k / 30k / 40k
workers: 16
device: 0
seed: 0
prefix: {prefix}
"""

ORDER_DESC = {
    "A": "published topical order",
    "B": "donor strength descending",
    "C": "donor strength ascending (exact reverse of B)",
}


def canonical(key):
    return dict((s[0], s[2]) for s in SOURCES)[key]


def plan():
    """Walk every order's rungs, deciding new-vs-reuse by source set."""
    # frozenset(sources) -> provenance label
    known = {}
    for rung, run in EXISTING_LADDER.items():
        known[frozenset(ORDERS["A"][:rung])] = f"order-A ladder: {run}"
    for key, run in SS_RUN.items():
        known.setdefault(frozenset([key]), f"single-source matrix: {run}")

    rows = []
    for order in ("A", "B", "C"):
        seq = ORDERS[order]
        for rung in range(1, 9):
            sources = seq[:rung]
            fs = frozenset(sources)
            prefix = f"nm_ladder_ord{order}_r{rung}"
            if fs in known:
                rows.append(dict(
                    order=order, rung=rung, added=seq[rung - 1], sources=sources,
                    status="reuse", run=known[fs], config="",
                ))
            else:
                known[fs] = f"order-{order} rung {rung}: {prefix}"
                rows.append(dict(
                    order=order, rung=rung, added=seq[rung - 1], sources=sources,
                    status="new", run=prefix, config=f"train_ord{order}_r{rung}.yaml",
                ))
    return rows


def write_config(out_dir, row):
    sources = row["sources"]
    text = CONFIG_TEMPLATE.format(
        title=f"NM ladder order-robustness -- order {row['order']} rung {row['rung']}",
        order=row["order"], order_desc=ORDER_DESC[row["order"]], rung=row["rung"],
        added=row["added"], added_canonical=canonical(row["added"]),
        sources_pretty=", ".join(canonical(s) for s in sources),
        root=ALL8_ROOT, graph=ALL8_GRAPH,
        subset=",".join(sources), prefix=row["run"],
    )
    (out_dir / row["config"]).write_text(text)


def write_gate_config(out_dir):
    """Order A rung 4 re-run through the subset knob -- the equivalence gate.

    Must reproduce the published nm_ladder_4src row within ~.01 before anything else
    runs. If it does not, the subset shortcut is invalid and every rung below is
    meaningless, so this is the one blocking check.
    """
    sources = ORDERS["A"][:4]
    text = CONFIG_TEMPLATE.format(
        title="EQUIVALENCE GATE -- order A rung 4 via the subset knob",
        order="A", order_desc="published topical order (gate re-run)", rung=4,
        added="covid_political", added_canonical="covid-political",
        sources_pretty=", ".join(canonical(s) for s in sources),
        root=ALL8_ROOT, graph=ALL8_GRAPH,
        subset=",".join(sources), prefix="nm_ladder_gate_ordA_r4",
    )
    text = text.replace(
        "# Trains on the all8 merged graph",
        "# GATE: this rung already exists as nm_ladder_4src, trained on a purpose-built\n"
        "# 4-source merge. Re-running it here through the subset knob tests the shortcut\n"
        "# against a known answer. Expect agreement within ~.01 on all 8 eval columns.\n"
        "# Trains on the all8 merged graph",
    )
    (out_dir / "train_gate_ordA_r4.yaml").write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    rows = plan()
    new = [r for r in rows if r["status"] == "new"]

    for order in ("A", "B", "C"):
        print(f"\nOrder {order} -- {ORDER_DESC[order]}")
        for r in [x for x in rows if x["order"] == order]:
            mark = "NEW " if r["status"] == "new" else "    "
            detail = r["run"] if r["status"] == "new" else f"reuse <- {r['run']}"
            print(f"  {mark}r{r['rung']}  +{r['added']:<18} ({len(r['sources'])} src)  {detail}")

    print(f"\n{len(new)} new training runs, {len(rows) - len(new)} reused, "
          f"{len(new) * 8} eval jobs.")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    for r in new:
        write_config(out_dir, r)
    write_gate_config(out_dir)

    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "rung", "added", "n_sources", "sources", "status",
                    "run_or_provenance", "config"])
        for r in rows:
            w.writerow([r["order"], r["rung"], r["added"], len(r["sources"]),
                        " ".join(r["sources"]), r["status"], r["run"], r["config"]])

    print(f"\nwrote {len(new)} configs + train_gate_ordA_r4.yaml + manifest.csv to {out_dir}")


if __name__ == "__main__":
    main()
