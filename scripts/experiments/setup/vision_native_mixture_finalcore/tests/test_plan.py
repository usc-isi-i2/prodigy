from scripts.experiments.setup.vision_native_mixture_finalcore.mixture_plan import (
    RUNGS,
    build_mixture_models,
)


def test_three_order_odd_rung_plan_is_deduplicated():
    models = build_mixture_models()
    assert len(models) == 13
    assert len({frozenset(model.sources) for model in models}) == 13
    assert sorted(len(model.sources) for model in models) == [1] * 3 + [3] * 3 + [5] * 3 + [7] * 3 + [9]
    all9 = next(model for model in models if model.model_id == "all9")
    assert len(all9.aliases) == 3
    assert RUNGS == (1, 3, 5, 7, 9)
