#!/usr/bin/env python3
"""Build the deduplicated three-order VISION native-mixture plan."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.experiments.setup.final_core.core_plan import ORDERS, build_models


RUNGS = (1, 3, 5, 7, 9)


@dataclass(frozen=True)
class MixtureModel:
    model_id: str
    sources: tuple[str, ...]
    aliases: tuple[str, ...]


def build_mixture_models() -> list[MixtureModel]:
    core_by_set = {frozenset(model.sources): model for model in build_models()}
    selected: dict[frozenset[str], MixtureModel] = {}
    for order_name, order in ORDERS.items():
        for rung in RUNGS:
            sources = tuple(order[:rung])
            key = frozenset(sources)
            alias = f"ladder:{order_name}:{rung}"
            if key in selected:
                prior = selected[key]
                selected[key] = MixtureModel(
                    prior.model_id, prior.sources, prior.aliases + (alias,)
                )
            else:
                core = core_by_set[key]
                selected[key] = MixtureModel(core.model_id, sources, (alias,))
    models = list(selected.values())
    if len(models) != 13:
        raise AssertionError(f"expected 13 unique mixture models, got {len(models)}")
    if sum(model.model_id == "all9" for model in models) != 1:
        raise AssertionError("all-nine mixture must be deduplicated")
    return models


def main() -> int:
    print("model_id\tn_sources\tsources\taliases")
    for model in build_mixture_models():
        print(
            f"{model.model_id}\t{len(model.sources)}\t{','.join(model.sources)}\t"
            f"{','.join(model.aliases)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
