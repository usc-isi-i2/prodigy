#!/usr/bin/env python3
"""Build the deduplicated 31-model PRODIGY final-core plan."""

from __future__ import annotations

from dataclasses import dataclass


SOURCES = (
    "ukr_rus",
    "covid",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk",
    "facebook_page_reference",
)
ORDERS = {
    "A": SOURCES,
    "B": (
        "ukr_rus_suspended", "election2020", "ukr_rus", "facebook_page_reference",
        "cp_hk", "covid_political", "covid", "midterm", "twibot20",
    ),
    "C": (
        "twibot20", "midterm", "covid", "covid_political", "cp_hk",
        "facebook_page_reference", "ukr_rus", "election2020", "ukr_rus_suspended",
    ),
}


@dataclass(frozen=True)
class CoreModel:
    model_id: str
    sources: tuple[str, ...]
    aliases: tuple[str, ...]


def build_models() -> list[CoreModel]:
    by_set: dict[frozenset[str], dict[str, object]] = {}
    for source in SOURCES:
        by_set[frozenset((source,))] = {
            "model_id": f"ss_{source}", "sources": (source,), "aliases": [f"specialist:{source}"]
        }
    for order_name, order in ORDERS.items():
        for rung in range(1, 10):
            key = frozenset(order[:rung])
            alias = f"ladder:{order_name}:{rung}"
            if key in by_set:
                by_set[key]["aliases"].append(alias)
            else:
                by_set[key] = {
                    "model_id": "all9" if rung == 9 else f"ord{order_name}_r{rung}",
                    "sources": tuple(order[:rung]),
                    "aliases": [alias],
                }
    models = [
        CoreModel(str(row["model_id"]), tuple(row["sources"]), tuple(row["aliases"]))
        for row in by_set.values()
    ]
    if len(models) != 31:
        raise AssertionError(f"expected 31 unique models, got {len(models)}")
    return models


def main() -> int:
    print("model_id\tn_sources\tsources\taliases")
    for model in build_models():
        print(f"{model.model_id}\t{len(model.sources)}\t{','.join(model.sources)}\t{','.join(model.aliases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
