#!/usr/bin/env python3
"""Generate the nine leave-one-source-out configs from the frozen base config."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from loo_plan import build_models


HERE = Path(__file__).resolve().parent


def render(base: str, model_id: str, sources: tuple[str, ...]) -> str:
    marker = "prefix: nmloo_base"
    if base.count(marker) != 1:
        raise ValueError(f"training.yaml must contain exactly one {marker!r}")
    text = base.replace(marker, f"prefix: {model_id}")
    return text.rstrip() + f'\nneighbor_sampling_source_subset: "{",".join(sources)}"\n'


def generate(output_dir: Path, *, replace: bool) -> None:
    if output_dir.exists():
        if not replace:
            raise FileExistsError(f"refusing existing config directory {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    base = (HERE / "training.yaml").read_text(encoding="utf-8")
    for index, model in enumerate(build_models()):
        path = output_dir / f"{index:02d}_{model.model_id}.yaml"
        path.write_text(render(base, model.model_id, model.sources), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "configs")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    generate(args.output_dir, replace=args.replace)
    print(f"wrote 9 configs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
