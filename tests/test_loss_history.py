import json

from mixture_scaling.loss_history import collect


def test_collect_preserves_overall_and_per_source_losses(tmp_path):
    run = tmp_path / "specialist_graph_s0"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"sources": ["graph"], "seed": 0}))
    (run / "summary.json").write_text(json.dumps({"status": "complete", "best_step": 250, "final_step": 250}))
    (run / "validation.jsonl").write_text(
        json.dumps(
            {
                "step": 250,
                "train_loss": 0.6,
                "validation_loss": 0.7,
                "per_source_validation_loss": {"graph": 0.7},
                "elapsed_seconds": 2.5,
            }
        )
        + "\n"
    )
    history, per_source = collect([("test", tmp_path)])
    assert history[0]["validation_loss"] == 0.7
    assert per_source[0]["validation_source"] == "graph"
    assert per_source[0]["validation_loss"] == 0.7
