"""Bounded ordered CPU gathers and CUDA transfers, preserving sampler RNG."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

from .node_only_transfer import DirectLinkBatch


class PrefetchLinks:
    def __init__(self, loaders, schedule, device, depth=8, workers=4):
        if depth < 0 or workers < 1:
            raise ValueError("invalid prefetch settings")
        self.loaders, self.schedule, self.device = loaders, iter(schedule), torch.device(device)
        self.depth = depth
        self.iterators = {}
        self.pending = deque()
        self.pool = ThreadPoolExecutor(max_workers=workers) if depth else None
        self.stream = torch.cuda.Stream(device=self.device) if depth and self.device.type == "cuda" else None
        self.ready = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.stream:
            self.stream.synchronize()
        self.pending.clear()
        self.ready = None

    def __iter__(self):
        return self

    def _sample(self, source):
        loader = self.loaders[source]
        if source not in self.iterators:
            self.iterators[source] = iter(loader.endpoint_batches())
        try:
            return next(self.iterators[source])
        except StopIteration:
            self.iterators[source] = iter(loader.endpoint_batches())
            return next(self.iterators[source])

    def _prepare(self, source, sampled):
        # CUDA current-device state is thread local. Pinning must not initialize GPU 0.
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        ids, edges, labels = sampled
        x = self.loaders[source].x
        values = ids if x.is_cuda else x[ids]
        if self.device.type == "cuda":
            values, edges, labels = values.pin_memory(), edges.pin_memory(), labels.pin_memory()
        return source, values, edges, labels

    def _fill(self):
        while len(self.pending) < self.depth:
            try:
                source = next(self.schedule)
            except StopIteration:
                break
            sampled = self._sample(source)
            self.pending.append(self.pool.submit(self._prepare, source, sampled))

    def _transfer(self, prepared):
        source, values, edges, labels = prepared
        values = values.to(self.device, non_blocking=True)
        x = self.loaders[source].x
        features = x[values] if x.is_cuda else values
        return source, DirectLinkBatch(features, edges.to(self.device, non_blocking=True),
                                       labels.to(self.device, non_blocking=True))

    def __next__(self):
        if not self.depth:
            source = next(self.schedule)
            return self._transfer(self._prepare(source, self._sample(source)))
        self._fill()
        if self.ready is None:
            if not self.pending:
                raise StopIteration
            prepared = self.pending.popleft().result()
            if self.stream:
                with torch.cuda.stream(self.stream):
                    self.ready = self._transfer(prepared)
            else:
                self.ready = self._transfer(prepared)
        result = self.ready
        self.ready = None
        if self.stream:
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(self.stream)
            for tensor in (result[1].x, result[1].edge_label_index, result[1].edge_label):
                tensor.record_stream(current)
        # Enqueue the next copy before returning; it overlaps the caller's compute.
        self._fill()
        if self.pending:
            prepared = self.pending.popleft().result()
            if self.stream:
                with torch.cuda.stream(self.stream):
                    self.ready = self._transfer(prepared)
            else:
                self.ready = self._transfer(prepared)
        return result
