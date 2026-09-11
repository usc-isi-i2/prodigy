"""Real two-source training benchmark; identical batches, optimizer and weights."""
import argparse
import json
import time
from pathlib import Path

import torch

from .config import load_config
from .lattice import load_shared_graph
from .node_only_transfer import build_model, make_direct_lp_loader, lp_loss, seed_everything
from .prefetch_links import PrefetchLinks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, choices=(0, 1, 2, 3), default=1)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--output", required=True)
    p.add_argument("--cache-root", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    config = load_config("configs/node_only_transfer.yaml")
    protocol = config["protocol"]
    graphs, features = {}, {}
    for source in ("ukr_rus_twitter", "covid19_twitter"):
        graphs[source] = load_shared_graph(source, config["graphs"][source]["path"], "lp",
                                          protocol, 0, Path(args.cache_root))
        features[source] = graphs[source].data.x.to(device) if source == "ukr_rus_twitter" else graphs[source].data.x
    results, reference = [], None
    for depth in (0, 8, 16):
        seed_everything(0)
        model = build_model("lp", protocol, device)
        opt = torch.optim.AdamW(model.parameters(), lr=protocol["node_mlp_lp_learning_rate"],
                               weight_decay=protocol["weight_decay"])
        loaders = {s: make_direct_lp_loader(graphs[s], protocol, False, 0, features[s]) for s in graphs}
        schedule = [list(graphs)[i % 2] for i in range(args.steps+20)]
        with PrefetchLinks(loaders, schedule, device, depth=depth, workers=4) as batches:
            for step, (source, batch) in enumerate(batches):
                if step == 20:
                    torch.cuda.synchronize(device)
                    start = time.monotonic()
                opt.zero_grad(set_to_none=True)
                loss = lp_loss(model, batch, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite benchmark loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip_norm"])
                opt.step()
                float(loss.detach())  # match training's metrics synchronization
            torch.cuda.synchronize(device)
            seconds = time.monotonic()-start
        state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if reference is None:
            reference = state
        max_diff = max((state[k]-reference[k]).abs().max().item() for k in state)
        assert max_diff == 0, f"prefetch changed weights: {max_diff}"
        row = {"prefetch_depth": depth, "steps": args.steps, "seconds": seconds,
               "updates_per_second": args.steps/seconds, "max_weight_difference": max_diff}
        results.append(row)
        print(json.dumps(row), flush=True)
        del model, opt, batch, loss
    Path(args.output).write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
