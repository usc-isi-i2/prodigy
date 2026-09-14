"""Atomic, inspectable checkpoint writes. RNG state does not imply sampler resume."""
import os
from pathlib import Path
import random
import subprocess
import tempfile
import numpy as np
import torch


def atomic_torch_save(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent);os.close(fd)
    try:
        torch.save(payload,tmp)
        with open(tmp,'rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def runtime_state():
    try:revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):revision=None
    return {'schema_version':2,'code_revision':revision,
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
                   'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else []},
            'exact_resume_supported':False,'resume_limitation':'Sampler iterators and outstanding prefetch batches are not serialized.'}
