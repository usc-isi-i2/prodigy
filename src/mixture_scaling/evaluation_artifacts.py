"""Versioned evaluation provenance and atomic score/JSON sidecars."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .binary_metrics import SCHEMA_VERSION


def fingerprint(path, content=False):
    path=Path(path).resolve();s=path.stat();result={'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
    if content:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        result['sha256']=h.hexdigest()
    return result


def signature(checkpoint,graph,split,seed,max_positives):
    return {'schema_version':SCHEMA_VERSION,'checkpoint':fingerprint(checkpoint,True),'graph':fingerprint(graph),
            'edge_split':fingerprint(split),'seed':seed,'max_positives':max_positives,
            'negative_kind':'degree_matched','validation_fraction':0.3}


def destination(primary):
    primary=Path(primary)
    if primary.exists() and json.loads(primary.read_text()).get('schema_version')!=SCHEMA_VERSION:
        return primary.with_name(primary.stem+'.metrics-v2.json')
    return primary


def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(payload,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def save_scores(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
