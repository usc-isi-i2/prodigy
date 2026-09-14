from collections import Counter

from mixture_scaling.node_mlp_ladder import ladder_rows, resident_sources, source_schedule


def test_ladder_has_nested_prefixes_and_equal_total_budget():
    rows = ladder_rows()
    assert len(rows) == 9
    for i, (_, sources) in enumerate(rows, 1):
        assert len(sources) == i
        if i > 1:
            assert sources[:-1] == rows[i-2][1]
        counts = Counter(source_schedule(sources, 2500))
        assert sum(counts.values()) == 2500
        assert max(counts.values()) - min(counts.values()) <= 1


def test_residency_respects_budget_and_keeps_smaller_sources():
    sizes = {"large": 66, "small_a": 8, "small_b": 7}
    assert resident_sources(sizes, 70) == {"small_a", "small_b"}
    assert resident_sources(sizes, 81) == set(sizes)
    assert resident_sources(sizes, 0) == set()


def test_tiny_training_persists_metrics_and_periodic_terminal_state(tmp_path, monkeypatch):
    import json
    import torch
    from types import SimpleNamespace
    from mixture_scaling.node_mlp_ladder import train_rung
    from mixture_scaling.config import load_config
    config=load_config('configs/node_only_transfer.yaml')
    config['protocol'].update(input_dim=4,hidden_dim=8,output_dim=8,ssl_batch_size=2,validation_batches=2)
    graph=SimpleNamespace(data=SimpleNamespace(x=torch.randn(8,4)),
        train_edges=torch.tensor([[0,1,2,3],[1,2,3,4]]),validation_edges=torch.tensor([[4,5],[5,6]]))
    args=SimpleNamespace(state_root=str(tmp_path),convergence=False,steps=4,seed=0,
        feature_budget_gib=0,minimum_steps_per_source=1,validation_every_per_source=2,
        patience=2,min_delta=0.,max_steps_per_source=4,checkpoint_interval=2,
        prefetch_depth=0,prefetch_workers=1,wandb_mode='disabled',wandb_project='test',wandb_group='test',log_interval=2)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(0,0))
    train_rung('tiny',['a'],{'a':graph},config,args,torch.device('cpu'))
    root=tmp_path/'lp/tiny'
    assert (root/'best.pt').exists() and (root/'checkpoints/step_2.pt').exists()
    latest=torch.load(root/'latest.pt',weights_only=False)
    assert latest['step']==4 and latest['optimizer']['state']
    report=json.loads((root/'validation/step_4.json').read_text())['reports']['a']
    assert 'roc_auc' in report and 'accuracy' in report['at_probability_0_5']
    history=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    assert any('train/accuracy' in row for row in history)
