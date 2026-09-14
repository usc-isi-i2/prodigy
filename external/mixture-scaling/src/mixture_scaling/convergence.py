"""Stop only after every source's fixed validation loss has plateaued."""
import math


class SourcePlateau:
    def __init__(self, sources, patience=10, min_delta=1e-4):
        self.best = {s: math.inf for s in sources}
        self.stale = dict.fromkeys(sources, 0)
        self.patience = patience
        self.min_delta = min_delta

    def update(self, losses, eligible):
        if losses.keys() != self.best.keys():
            raise ValueError("validation source set changed")
        for source, loss in losses.items():
            if not math.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            if self.best[source] - loss > self.min_delta:
                self.best[source] = loss
                self.stale[source] = 0
            elif eligible:
                self.stale[source] += 1
        return eligible and all(n >= self.patience for n in self.stale.values())
