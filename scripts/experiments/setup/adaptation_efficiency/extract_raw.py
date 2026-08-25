#!/usr/bin/env python3
"""Create raw-feature logistic and MLP caches on the shared labeled nodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import FeatureCache, save_feature_cache
from .targets import graph_field, labeled_nodes, load_graph, load_labels, selected_targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", default="covid_political,election2020,ukr_rus_suspended,twibot20")
    args = parser.parse_args()
    for target in selected_targets(args.targets):
        graph = load_graph(target)
        labels = load_labels(graph, target.label_key)
        nodes = labeled_nodes(labels)
        features = graph_field(graph, "x")[nodes].float().cpu().numpy()
        for model_id in ("raw_logistic", "raw_mlp"):
            save_feature_cache(
                args.output_root / model_id / f"{target.name}.npz",
                FeatureCache(
                    model_id=model_id,
                    target=target.name,
                    features=features,
                    labels=labels,
                    node_ids=nodes,
                    metadata={
                        "source": str(target.graph),
                        "label_key": target.label_key,
                        "representation": "raw_node_features",
                    },
                ),
            )
        print(f"{target.name}: labeled_nodes={nodes.size} raw_dim={features.shape[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

