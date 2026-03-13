"""
Attach user embeddings to mention_graph.pt as node features (data.x).

The embedding is concatenated with the existing stat features in data.x so
both signals are preserved:
    data.x = [stat_features | text_embedding]   shape: (N, F_stat + D_emb)

Nodes without a post embedding get a zero vector in the embedding part.

Usage:
    python attach_embeddings.py \
        --graph      mention_graph.pt \
        --embeddings user_embeddings_minilm.pt \
        --out        mention_graph_minilm.pt          # default
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph",      default="mention_graph.pt")
    parser.add_argument("--embeddings", default="user_embeddings_minilm.pt")
    parser.add_argument("--pool",       default="meanpool", choices=["meanpool", "maxpool"],
                        help="Which pooling to use as node features")
    parser.add_argument("--out",        default="mention_graph_minilm.pt")
    args = parser.parse_args()

    # ── Load graph ────────────────────────────────────────────────────────────
    print(f"Loading graph from {args.graph}...")
    ckpt = torch.load(args.graph, map_location="cpu")
    data, h2i, handles = ckpt["data"], ckpt["h2i"], ckpt["handles"]
    num_nodes = data.x.shape[0]
    stat_x = data.x                                          # (N, F_stat)
    stat_feature_names = list(getattr(data, "feature_names", []))
    print(f"  {num_nodes} nodes, {stat_x.shape[1]} stat features: {stat_feature_names}")

    # ── Load embeddings ───────────────────────────────────────────────────────
    print(f"Loading embeddings from {args.embeddings}...")
    emb_ckpt = torch.load(args.embeddings, map_location="cpu")
    emb_handles = emb_ckpt["handles"]          # list[str]
    emb_matrix  = emb_ckpt[args.pool]          # (M, D)
    D = emb_matrix.shape[1]
    handle_to_emb = dict(zip(emb_handles, emb_matrix))
    print(f"  {len(emb_handles)} users with embeddings, dim={D}")

    # ── Align embeddings to graph node order ──────────────────────────────────
    emb_x = torch.zeros(num_nodes, D)
    matched = 0
    for handle, idx in h2i.items():
        if idx >= num_nodes:
            continue
        emb = handle_to_emb.get(handle)
        if emb is not None:
            emb_x[idx] = emb
            matched += 1

    unmatched_emb = len(handle_to_emb) - matched
    print(f"  Matched {matched}/{num_nodes} nodes ({matched/num_nodes*100:.1f}%)")
    print(f"  {num_nodes - matched} nodes padded with zeros (in graph, no embedding)")
    print(f"  {unmatched_emb} embeddings not matched to any graph node (have posts but not in graph)")

    # ── Concatenate stat features + embedding ─────────────────────────────────
    data.x = torch.cat([stat_x, emb_x], dim=1)              # (N, F_stat + D)
    data.feature_names = stat_feature_names + [f"emb_{i}" for i in range(D)]
    print(f"  Combined x shape: {data.x.shape}  "
          f"(stat={stat_x.shape[1]} + emb={D})")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({"data": data, "h2i": h2i, "handles": handles}, args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
