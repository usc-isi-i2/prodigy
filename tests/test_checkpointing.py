import torch
import pytest
from mixture_scaling.checkpointing import atomic_torch_save
from mixture_scaling.node_only_transfer import save_checkpoint


def test_checkpoint_preserves_optimizer_rng_and_restorable_parameters(tmp_path):
    model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters());model(torch.ones(1,3)).sum().backward();optimizer.step()
    path=tmp_path/'best.pt';before=torch.get_rng_state().clone();save_checkpoint(path,model,optimizer,7,{'source':'test'})
    payload=torch.load(path,weights_only=False)
    assert payload['step']==7 and payload['optimizer']['state']
    assert torch.equal(payload['rng']['torch_cpu'],before)
    assert payload['exact_resume_supported'] is False
    restored=torch.nn.Linear(3,2);restored.load_state_dict(payload['model'])
    assert torch.equal(model.weight,restored.weight)


def test_failed_atomic_write_keeps_previous_checkpoint(tmp_path,monkeypatch):
    path=tmp_path/'latest.pt';atomic_torch_save({'step':1},path)
    def fail(*args,**kwargs):raise OSError('disk failure')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError):atomic_torch_save({'step':2},path)
    assert torch.load(path,weights_only=False)['step']==1
    assert list(tmp_path.iterdir())==[path]
