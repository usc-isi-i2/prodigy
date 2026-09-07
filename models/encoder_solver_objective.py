"""Parameter-free, support-fitted ridge objective for the bounded isolation study."""
import torch
import torch.nn.functional as F


def configure_objective(params):
    mode = params.get("encoder_solver_objective", "native")
    if mode not in {"native", "joint", "isolated", "ridge_only"}:
        raise ValueError(f"Unknown encoder_solver_objective: {mode}")
    if mode != "native":
        checks = {
            "layers=S,U,M": params.get("layers") == "S,U,M",
            "neighbor_matching objective": params.get("task_name") == "neighbor_matching",
            "multiway episodes": params.get("n_way", 0) > 1,
            "no auxiliary regression": not params.get("attr_regression_weight", 0),
            "no dropout": not params.get("dropout", 0) and not params.get("text_features_dropout", 0),
            "no zero-shot": not params.get("zero_shot", False),
            "one metagraph layer": params.get("meta_n_layer", 1) == 1,
            "no final back propagation": not params.get("has_final_back", False),
            "no skip path": not params.get("skip_path", False),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError("Encoder/solver isolation requires " + ", ".join(failed))
    params["encoder_solver_objective"] = mode
    params["encoder_solver_effective"] = {
        "mode": mode, "ridge_lambda": 1.0, "ridge_logit_scale": 1.0,
        "ridge_intercept": False, "ridge_feature_normalization": "row_l2",
        "ridge_target": "support_one_hot", "ridge_loss": "query_cross_entropy",
        "native_loss_weight": 0.0 if mode == "ridge_only" else 1.0,
        "ridge_loss_weight": 0.0 if mode == "native" else 1.0,
        "detach_u1_for_native": mode in {"isolated", "ridge_only"},
    }
    return mode


def ridge_query_loss(u1, labels, edge_index, query_mask):
    """Fit each actual metagraph task independently; never fit query labels."""
    n, ways = labels.shape
    if ways < 2 or u1.shape[0] != n:
        raise ValueError("Ridge requires multiway example-aligned labels")
    edges = edge_index.reshape(2, n, ways)
    expected = torch.arange(n, device=u1.device)[:, None].expand(n, ways)
    if not torch.equal(edges[0], expected):
        raise ValueError("Unsupported metagraph example ordering")
    task_start = edges[1, :, 0]
    if not torch.equal(edges[1], task_start[:, None] + torch.arange(ways, device=u1.device)):
        raise ValueError("Unsupported metagraph class ordering")
    mask = query_mask.reshape(n, ways).bool()
    if not torch.equal(mask, mask[:, :1].expand_as(mask)):
        raise ValueError("Inconsistent query role within example")
    q = mask[:, 0]
    z = F.normalize(u1, dim=-1)
    losses = []
    for task in torch.unique(task_start):
        rows = task_start == task
        support, query = rows & ~q, rows & q
        if not support.any() or not query.any() or not (labels[support].sum(0) > 0).all():
            raise ValueError("Each task needs support for every class and queries")
        xs = z[support]
        # Dual solve is exact ridge with lambda=1 and no intercept.
        gram = xs @ xs.T + torch.eye(xs.shape[0], device=xs.device, dtype=xs.dtype)
        coef = torch.linalg.solve(gram, labels[support].to(xs.dtype))
        logits = z[query] @ xs.T @ coef
        losses.append(F.cross_entropy(logits, labels[query].argmax(-1), reduction="sum"))
    return torch.stack(losses).sum() / q.sum()
