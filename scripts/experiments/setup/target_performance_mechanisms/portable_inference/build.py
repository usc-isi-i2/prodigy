"""Build a self-contained inference candidate from explicit, auditable source units."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SETUP = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path, names):
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    chunks = []
    for name in names:
        n = nodes[name]
        start = min([n.lineno] + [d.lineno for d in n.decorator_list])
        chunks.append(''.join(lines[start-1:n.end_lineno]))
    return '\n\n'.join(chunks)


def build(output):
    if output.exists():
        raise ValueError('Use a new package directory')
    output.mkdir(parents=True)
    package = output / 'graph_role'
    (package / 'models').mkdir(parents=True)
    provenance = []
    for name in ('layer_classes', 'general_gnn', 'multilayer_gnn', 'gnn_with_edge_attr', 'metaGNN', 'supernode_propagation_layers'):
        original = REPO / 'models' / (name+'.py')
        destination = package / 'models' / (name+'.py')
        destination.write_text(original.read_text().replace('from models.', 'from graph_role.models.'))
        provenance.append({'source': str(original.relative_to(REPO)), 'source_sha256': sha(original),
                           'destination': str(destination.relative_to(output)), 'transform': 'import namespace only'})
    selections = [
        ('replay.py', ['clone_batch', 'tensor_hash', 'batch_hash', 'trace_stages', 'episode_probe'],
         'import hashlib\nimport json\nfrom contextlib import contextmanager\nimport torch\nimport torch.nn.functional as F\nfrom graph_role.models.layer_classes import BackgroundGNNLayer, SupernodeAggrLayer, MetagraphLayer\n', 'replay.py'),
        ('role_context.py', ['query_mask'], '', 'role_context.py'),
        ('episode_cardinality.py', ['cached_meta_forward'], 'import torch\n', 'episode_cardinality.py'),
        ('verify_member_training.py', ['model_digest'], 'import hashlib\nimport json\nimport torch\n', 'digest.py'),
        ('analyze_mixture_predictions.py', ['evaluate_probabilities', 'evaluate_logits'],
         'import torch\nimport torch.nn.functional as F\nfrom sklearn.metrics import accuracy_score, f1_score, roc_auc_score\n', 'metrics.py'),
    ]
    for filename, names, prelude, dest in selections:
        original = SETUP / filename
        selected = extract(original, names)
        (package / dest).write_text(prelude+'\n'+selected+'\n')
        provenance.append({'source': str(original.relative_to(REPO)), 'source_sha256': sha(original),
                           'symbols': names, 'selected_sha256': hashlib.sha256(selected.encode()).hexdigest(),
                           'destination': 'graph_role/'+dest, 'transform': 'verbatim symbols; explicit minimal imports'})
    for name in ('role_topology.py', 'message_content.py', 'support_dose.py'):
        original = SETUP / name
        (package / name).write_text(original.read_text().replace('from models.', 'from graph_role.models.'))
        provenance.append({'source': str(original.relative_to(REPO)), 'source_sha256': sha(original),
                           'destination': 'graph_role/'+name, 'transform': 'import namespace only'})
    for original in sorted((HERE / 'template').rglob('*')):
        if original.is_file() and '__pycache__' not in original.parts:
            dest = output / original.relative_to(HERE / 'template')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, dest)
    for dest in (package / '__init__.py', package / 'models/__init__.py'):
        dest.write_text('')
    (output / 'SOURCE_PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    forbidden = ('/Users/', '/dataMeR1/', '/home/mhchu/', 'wandb', 'from experiments.', 'from scripts.')
    files = sorted(p for p in output.rglob('*') if p.is_file())
    for p in files:
        if p.suffix == '.py' and any(s in p.read_text() for s in forbidden):
            raise ValueError('Unportable import/path: '+str(p))
    manifest = {'format': 1, 'scope': 'frozen_inference_code_and_synthetic_fixtures',
                'release_status': 'local_candidate_not_uploaded',
                'sha256': {str(p.relative_to(output)): sha(p) for p in files}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise ValueError('Archive already exists')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                info = zipfile.ZipInfo(str(Path(output.name)/p.relative_to(output)), (2026, 9, 7, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, p.read_bytes())
    print(json.dumps({'directory': str(output), 'archive': str(archive), 'sha256': sha(archive),
                      'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                      'files': len(files)+1, 'public_upload': False}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    build(p.parse_args().output.resolve())
