"""Audit assigned-anchor errors against cached canonical positive edges.

No full graph loading, model inference, or new samples. Observed positives are
incomplete unless the complete edge-only HK cache is explicitly supplied:
absence from the observed set is NOT evidence that an edge is absent.
The alternative-positive score is descriptive, not a replacement benchmark.
"""
import argparse
import collections
import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_reference(storage, offset, size, stride, *unused):
    return dict(storage=storage, offset=offset, size=size, stride=stride)


class EdgeMetadata(pickle.Unpickler):
    """Read the trusted edge-only cache without importing torch or graph code."""
    def persistent_load(self, pid):
        assert pid[0] == "storage" and pid[1] is int
        return dict(key=pid[2], count=pid[4])

    def find_class(self, module, name):
        if module == "torch._utils" and name in {"_rebuild_tensor", "_rebuild_tensor_v2"}:
            return tensor_reference
        if module == "torch" and name == "LongStorage":
            return int
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise ValueError((module, name))


def read_hk_views(path, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    assert digest(path) == receipt["canonical_views_sha256"]
    views = {}
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        with archive.open(name) as stream:
            metadata = EdgeMetadata(stream).load()
        prefix = name[:-len("data.pkl")]
        assert metadata["offset"] == 34148422
        for key, ref in metadata["views"].items():
            raw = archive.read(prefix + "data/" + str(ref["storage"]["key"]))
            assert len(raw) == ref["storage"]["count"] * 8
            view = np.ndarray(shape=ref["size"], dtype="<i8", buffer=raw,
                offset=ref["offset"] * 8, strides=tuple(v * 8 for v in ref["stride"]))
            assert view.shape == (2, receipt["canonical_view_edges"][key])
            assert (view >= 0).all() and (view < 333800).all()
            views[key] = {tuple(sorted(map(int, pair))) for pair in view.T}
    assert not (views["train"] & views["test"])
    assert not (views["validation"] & views["test"])
    assert not (views["train"] & views["validation"])
    return views, {"sha256": digest(path), "receipt_sha256": digest(receipt_path),
        "undirected_pair_counts": {k: len(v) for k, v in views.items()},
        "offset": metadata["offset"], "source_artifact": metadata["source_artifact"]}


def rates(d):
    out = {"occurrences": len(d), "queries": int(d["query"].nunique())}
    if not len(d):
        return out
    for head in ["native", "encoded", "prototype"]:
        correct = d[head + "_correct"]
        alternative = d[head + "_alternative_positive"]
        error = ~correct
        out[head] = {
            "correct": int(correct.sum()),
            "errors": int(error.sum()),
            "accuracy": float(correct.mean()),
            "known_alternative_positive_errors": int(alternative.sum()),
            "fraction_errors_known_alternative_positive": (
                float(alternative.sum() / error.sum()) if error.sum() else None
            ),
            "assigned_or_known_positive_rate": float((correct | alternative).mean()),
            "node_weighted_accuracy": float(d.groupby("query")[head + "_correct"].mean().mean()),
        }
        edge_column = head + "_predicted_edge_split"
        if edge_column in d:
            out[head]["error_predicted_edge_split"] = d.loc[error, edge_column].value_counts().to_dict()
            out[head]["assigned_or_any_graph_neighbor_rate"] = float(d[edge_column].ne("absent").mean())
    out["known_candidate_count_mean"] = float(d.known_candidate_count.mean())
    out["known_multi_candidate_occurrences"] = int(d.known_candidate_count.gt(1).sum())
    out["query_also_rival_support_occurrences"] = int(d.query_rival_support.sum())
    out["query_also_rival_query_occurrences"] = int(d.query_rival_query.sum())
    out["mean_uniform_candidate_known_positive_rate"] = float(d.known_candidate_count.mean() / 30)
    if "all_edge_candidate_count" in d:
        out["all_edge_candidate_count_mean"] = float(d.all_edge_candidate_count.mean())
        out["mean_uniform_candidate_any_graph_positive_rate"] = float(d.all_edge_candidate_count.mean() / 30)
    native_errors = ~d.native_correct
    nonalternative = native_errors & ~d.native_alternative_positive
    out["native_errors_without_observed_positive_match"] = int(nonalternative.sum())
    out["native_errors_without_observed_positive_match_encoded_correct"] = int(
        (nonalternative & d.encoded_correct).sum())
    out["native_errors_without_observed_positive_match_raw_groups"] = (
        d.loc[nonalternative, "raw_group"].value_counts().to_dict())
    return out


def analyze(reference_root, stage_root, geometry_root, hk_views=None, hk_views_receipt=None):
    provenance_root = Path(__file__).parent / "data/canonical_split"
    previous = json.loads((provenance_root / "nm_source_stages.json").read_text())
    result = {
        "protocol": "canonical_observed_test_membership_accounting_v1",
        "scope": "Observed test support/query pairs only; incomplete adjacency; undirected membership.",
        "source_stage_aggregate_sha256": digest(provenance_root / "nm_source_stages.json"),
        "targets": {},
    }
    if hk_views is not None:
        assert hk_views_receipt is not None
        complete_hk_views, full_receipt = read_hk_views(hk_views, hk_views_receipt)
        result["hk_edge_cache"] = full_receipt
        result["scope"] = "HK: complete static_test adjacency. Ukraine: incomplete observed canonical test pairs. Undirected membership; unchanged predictions."
    for target, name, expected_pairs in [("cp_hk", "hk", 42023), ("ukr_rus", "ukr", 105899)]:
        rpath = reference_root / (name + ".tsv")
        spath = stage_root / target / "stage_predictions_private.csv"
        gpath = geometry_root / target / "query_geometry_private.csv"
        reference = pd.read_csv(rpath, sep="\t")
        d = reference.loc[reference.split.eq("test")].copy()
        s = pd.read_csv(spath)
        g = pd.read_csv(gpath)
        receipt = previous["receipt"]["targets"][target]
        assert digest(rpath) == receipt["reference_sha256"]
        assert digest(spath) == receipt["csv_sha256"]
        assert digest(gpath) == previous["targets"][target]["geometry_sha256"]
        assert len(d) == len(g) == 61440 and len(s) == 122880
        assert not d.duplicated(["episode", "sample"]).any()
        assert not s.duplicated(["episode", "sample", "model"]).any()
        d["class_slot"] = (d["sample"].astype(int) % 210) // 7
        d["frequency"] = d.groupby("query")["query"].transform("size")
        assert d.groupby(["episode", "class_slot"]).size().eq(4).all()
        support_cols = ["true_support_" + str(i) for i in range(3)]
        assert d.groupby(["episode", "class_slot"])[support_cols + ["anchor"]].nunique().eq(1).all().all()
        classes = d.drop_duplicates(["episode", "class_slot"])
        ordered = set(zip(d.anchor.astype(int), d["query"].astype(int)))
        for col in support_cols:
            ordered.update(zip(classes.anchor.astype(int), classes[col].astype(int)))
        assert len(ordered) == expected_pairs, (target, len(ordered), expected_pairs)
        membership_pairs = ordered
        if hk_views is not None and name == "hk":
            membership_pairs = complete_hk_views["test"]
            assert {tuple(sorted(pair)) for pair in ordered}.issubset(membership_pairs)
        adjacency = {}
        for anchor, member in membership_pairs:
            adjacency.setdefault(anchor, set()).add(member)
            adjacency.setdefault(member, set()).add(anchor)
        if hk_views is not None and name == "hk":
            full_adjacency = {}
            for pairs in complete_hk_views.values():
                for anchor, member in pairs:
                    full_adjacency.setdefault(anchor, set()).add(member)
                    full_adjacency.setdefault(member, set()).add(anchor)
        labels = classes.set_index(["episode", "class_slot"]).anchor
        for head in ["native", "encoded", "prototype"]:
            index = pd.MultiIndex.from_arrays([s.episode, s[head + "_prediction"].astype(int)])
            s[head + "_anchor"] = labels.loc[index].to_numpy()
            assert s[head + "_correct"].astype(bool).eq(s[head + "_anchor"].eq(s.anchor)).all()
            s[head + "_correct"] = s[head + "_correct"].astype(bool)
            s[head + "_alternative_positive"] = [
                (not correct) and (int(predicted) in adjacency[int(query)])
                for query, predicted, correct in zip(s["query"], s[head + "_anchor"], s[head + "_correct"])
            ]
            if hk_views is not None and name == "hk":
                s[head + "_predicted_edge_split"] = [
                    next((split for split, pairs in complete_hk_views.items()
                          if tuple(sorted((int(query), int(predicted)))) in pairs), "absent")
                    for query, predicted in zip(s["query"], s[head + "_anchor"])
                ]
                assert s.loc[s[head + "_correct"], head + "_predicted_edge_split"].eq("test").all()
        flags, class_support_flags = [], []
        for episode, episode_rows in d.groupby("episode", sort=False):
            c = classes[classes.episode.eq(episode)].sort_values("class_slot")
            assert len(c) == 30 and c.class_slot.tolist() == list(range(30))
            anchors = set(map(int, c.anchor))
            assert len(anchors) == 30
            support_memberships = {}
            for row in c.itertuples(index=False):
                for col in support_cols:
                    support_memberships.setdefault(int(getattr(row, col)), set()).add(int(row.anchor))
            query_memberships = episode_rows.groupby("query").anchor.agg(set).to_dict()
            for row in episode_rows.itertuples(index=False):
                known = adjacency[int(row.query)] & anchors
                assert int(row.anchor) in known
                flags.append({
                    "episode": episode, "sample": row.sample,
                    "known_candidate_count": len(known),
                    "query_rival_support": bool(support_memberships.get(int(row.query), set()) - {int(row.anchor)}),
                    "query_rival_query": bool(query_memberships[int(row.query)] - {int(row.anchor)}),
                })
                if hk_views is not None and name == "hk":
                    flags[-1]["all_edge_candidate_count"] = len(full_adjacency[int(row.query)] & anchors)
            for row in c.itertuples(index=False):
                slots = [int(getattr(row, col)) for col in support_cols]
                alternative_counts = [len((adjacency[node] & anchors) - {int(row.anchor)}) for node in slots]
                class_support_flags.append({
                    "episode": episode, "anchor": int(row.anchor),
                    "supports_with_known_rival_membership": sum(n > 0 for n in alternative_counts),
                    "negative_support_edges_with_observed_positive_membership": sum(alternative_counts),
                })
        flags = pd.DataFrame(flags)
        class_support_flags = pd.DataFrame(class_support_flags)
        common = d[["episode", "sample", "query", "anchor", "frequency", "query_degree", "hk_correct", "ukr_correct", "hk_pred", "ukr_pred"]]
        common = common.merge(flags, on=["episode", "sample"], validate="one_to_one")
        geom = g[["episode", "sample", "query", "anchor", "full_mean_all_valid", "full_mean_margin"]]
        common = common.merge(geom, on=["episode", "sample", "query", "anchor"], validate="one_to_one")
        common["raw_group"] = np.where(~common.full_mean_all_valid, "undefined",
            np.where(common.full_mean_margin > 1e-7, "true_closest",
            np.where(common.full_mean_margin < -1e-7, "rival_closer", "tied")))
        common["frequency_group"] = pd.cut(common.frequency, [0, 1, 4, 19, np.inf], labels=["1", "2-4", "5-19", "20+"])
        out = {
            "membership_source": "complete_static_test" if hk_views is not None and name == "hk" else "observed_test_pairs_only",
            "input_hashes": {"reference": digest(rpath), "stages": digest(spath), "geometry": digest(gpath)},
            "observed_ordered_test_pairs": len(ordered),
            "observed_undirected_test_pairs": len({tuple(sorted(pair)) for pair in ordered}),
            "known_candidate_count_distribution": {int(k): int(v) for k, v in common.known_candidate_count.value_counts().sort_index().items()},
            "class_supports": {
                "classes": len(class_support_flags),
                "classes_with_known_rival_support_membership": int(class_support_flags.supports_with_known_rival_membership.gt(0).sum()),
                "support_occurrences": 46080,
                "support_occurrences_with_known_rival_membership": int(class_support_flags.supports_with_known_rival_membership.sum()),
                "negative_support_edges": 46080 * 29,
                "negative_support_edges_with_observed_positive_membership": int(class_support_flags.negative_support_edges_with_observed_positive_membership.sum()),
            },
            "models": {},
        }
        for model, frame in s.groupby("model"):
            frame = frame.merge(common, on=["episode", "sample", "query", "anchor"], validate="one_to_one")
            assert frame.native_correct.eq(frame[model + "_correct"].astype(bool)).all()
            assert frame.native_anchor.eq(frame[model + "_pred"]).all()
            m = {"all": rates(frame), "by_frequency": {str(k): rates(v) for k, v in frame.groupby("frequency_group", observed=True)}}
            for flag in ["query_rival_support", "query_rival_query"]:
                m["by_" + flag] = {str(k): rates(v) for k, v in frame.groupby(flag)}
            m["by_known_candidate_count"] = {str(k): rates(v) for k, v in frame.groupby("known_candidate_count")}
            class_rows = frame.groupby(["episode", "anchor"]).agg(
                original_correct=("native_correct", "sum"),
                known_alternative_positive_errors=("native_alternative_positive", "sum"),
            ).reset_index().merge(class_support_flags, on=["episode", "anchor"], validate="one_to_one")
            wrong_classes = class_rows[class_rows.original_correct.eq(0)]
            m["classes"] = {
                "all_four_wrong": len(wrong_classes),
                "all_four_wrong_without_any_known_positive_prediction": int(wrong_classes.known_alternative_positive_errors.eq(0).sum()),
                "all_four_wrong_with_any_known_positive_prediction": int(wrong_classes.known_alternative_positive_errors.gt(0).sum()),
                "all_four_wrong_with_all_four_known_positive_predictions": int(wrong_classes.known_alternative_positive_errors.eq(4).sum()),
            }
            m["class_support_membership_strata"] = {}
            for exposed in [False, True]:
                rows = class_rows[class_rows.supports_with_known_rival_membership.gt(0).eq(exposed)]
                m["class_support_membership_strata"][str(exposed)] = {
                    "classes": len(rows),
                    "native_accuracy": float(rows.original_correct.sum() / (len(rows) * 4)),
                    "all_four_wrong_fraction": float(rows.original_correct.eq(0).mean()),
                }
            out["models"][model] = m
        result["targets"][target] = out
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--stage-root", type=Path, required=True)
    p.add_argument("--geometry-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hk-views", type=Path)
    p.add_argument("--hk-views-receipt", type=Path)
    args = p.parse_args()
    result = analyze(args.reference_root, args.stage_root, args.geometry_root, args.hk_views, args.hk_views_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for target, name in [("cp_hk", "hk"), ("ukr_rus", "ukr")]:
        print(target, json.dumps(result["targets"][target]["models"][name]["all"], indent=2))
