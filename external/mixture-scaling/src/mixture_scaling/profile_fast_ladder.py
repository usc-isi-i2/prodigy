"""Short isolated replay of the real fast pipeline, with live-worker auto-resume."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

from .config import load_config
from .lattice import SOURCE_ORDER
from .node_only_transfer import build_model, lp_loss, make_direct_lp_loader, seed_everything
from .prefetch_links import PrefetchLinks


class TimedPrefetch(PrefetchLinks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_seconds = 0
        self.prepare_seconds = defaultdict(float)

    def _sample(self, source):
        start = time.perf_counter()
        result = super()._sample(source)
        self.sample_seconds += time.perf_counter() - start
        return result

    def _prepare(self, source, sampled):
        start = time.perf_counter()
        result = super()._prepare(source, sampled)
        self.prepare_seconds[source] += time.perf_counter() - start
        return result


@contextmanager
def pause_owned_worker(pid, device):
    tokens = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    if ("mixture_scaling.node_mlp_ladder" not in tokens or "--device" not in tokens
            or tokens[tokens.index("--device")+1] != str(device)):
        raise RuntimeError("refusing to pause an unrelated process")
    # Separate watchdog guarantees resumption even if the profiling process is killed.
    watchdog = subprocess.Popen([sys.executable, "-c",
        "import os,signal,time; time.sleep(30); os.kill(int(__import__('sys').argv[1]),signal.SIGCONT)",str(pid)])
    os.kill(pid, signal.SIGSTOP)
    try:
        yield
    finally:
        os.kill(pid, signal.SIGCONT)
        watchdog.terminate()
        watchdog.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device",type=int,choices=(0,1,2,3),required=True)
    p.add_argument("--pause-pid",type=int,required=True)
    p.add_argument("--rung",type=int,default=9)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    device = torch.device(f"cuda:{a.device}")
    protocol = load_config("configs/node_only_transfer.yaml")
    cfg = protocol["protocol"]
    sources = SOURCE_ORDER[:a.rung]
    graphs, features = {}, {}
    for source in sources:
        # Use the same deserializer and CPU feature representation as production.
        raw = torch.load(protocol["graphs"][source]["path"],map_location="cpu",weights_only=False)
        raw = raw if isinstance(raw,dict) else raw.to_dict()
        split = torch.load(a.cache_root/f"{source}_edge_split_s0.pt",map_location="cpu",weights_only=False)
        graphs[source]=SimpleNamespace(data=SimpleNamespace(x=raw["x"].float()),
            train_edges=split["train_edges"],validation_edges=split["validation_edges"])
        features[source] = graphs[source].data.x
    free,_=torch.cuda.mem_get_info(device)
    needed=sum(x.numel()*x.element_size() for s,x in features.items() if s!="covid19_twitter")
    if needed > free-5*2**30:
        raise RuntimeError("insufficient headroom for an isolated replay")
    for source in sources:
        if source!="covid19_twitter":
            features[source]=features[source].to(device)
    seed_everything(0)
    model=build_model("lp",cfg,device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["node_mlp_lp_learning_rate"],weight_decay=cfg["weight_decay"])
    loaders={s:make_direct_lp_loader(graphs[s],cfg,False,0,features[s]) for s in sources}
    phase_seconds=defaultdict(float)
    events=[]
    steps=180
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with TimedPrefetch(loaders,[sources[i%len(sources)] for i in range(steps+36)],device,depth=8,workers=4) as batches:
        # Prefill CPU futures and warm source pages before the brief pause.
        def update(measure):
            start=time.perf_counter()
            source,batch=next(batches)
            if measure:phase_seconds["next_batch"]+=time.perf_counter()-start
            for name,fn in [
                ("zero_grad",lambda:opt.zero_grad(set_to_none=True)),
                ("forward",lambda:lp_loss(model,batch,device)),
            ]:
                start=time.perf_counter(); result=fn()
                if name=="forward": loss=result
                if measure:phase_seconds[name]+=time.perf_counter()-start
            start=time.perf_counter()
            if not torch.isfinite(loss):raise FloatingPointError("non-finite loss")
            if measure:phase_seconds["finite_check_sync"]+=time.perf_counter()-start
            start=time.perf_counter();loss.backward()
            if measure:phase_seconds["backward_dispatch"]+=time.perf_counter()-start
            start=time.perf_counter();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"])
            if measure:phase_seconds["gradient_clip_dispatch"]+=time.perf_counter()-start
            start=time.perf_counter();opt.step()
            if measure:phase_seconds["optimizer_dispatch"]+=time.perf_counter()-start
            start=time.perf_counter();float(loss.detach())
            if measure:phase_seconds["scalar_log_read"]+=time.perf_counter()-start
        for _ in range(36):update(False)
        sample_before=batches.sample_seconds
        prep_before=dict(batches.prepare_seconds)
        with pause_owned_worker(a.pause_pid,a.device):
            torch.cuda.synchronize(device)
            started=time.perf_counter()
            for _ in range(steps):update(True)
            torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-started
        row={"steps":steps,"wall_seconds":elapsed,"updates_per_second":steps/elapsed,
            "host_phase_seconds":dict(phase_seconds),
            "host_phase_percent":{k:100*v/elapsed for k,v in phase_seconds.items()},
            "sampling_seconds_in_next_batch":batches.sample_seconds-sample_before,
            "parallel_prepare_worker_seconds":{s:v-prep_before.get(s,0) for s,v in batches.prepare_seconds.items()},
            "limitations":"Short real-data replay with the production CPU feature representation; host timings include dispatch and waits, not pure GPU kernel durations. Validation/W&B excluded. Other GPUs remain active."}
    a.output.write_text(json.dumps(row,indent=2)+"\n")
    print(json.dumps(row,indent=2),flush=True)


if __name__=="__main__":main()
