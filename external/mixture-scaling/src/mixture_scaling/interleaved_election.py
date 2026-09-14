"""Extend the completed eight-graph sweep, reusing its 28 selected models."""
import argparse
import json
from pathlib import Path
from . import interleaved_mlp_pairs as m
from .lattice import SOURCE_ORDER

REUSE_ROOT=Path('/dataMeR1/phil/gfm/mixture-scaling/state/interleaved_mlp_pairs_noelec_s0')


def reuse_completed(root,reuse_root=REUSE_ROOT):
    """Read-only links to complete runs; new Election pairs get new directories."""
    root=Path(root);reuse_root=Path(reuse_root)
    if root.resolve()==reuse_root.resolve():raise ValueError('extension needs a separate output root')
    receipt=json.loads((reuse_root/'results/COMPLETE.json').read_text())
    if receipt.get('status')!='complete' or receipt.get('interleaved_models')!=28 or receipt.get('evaluation_cells')!=224:
        raise ValueError('original sweep is incomplete')
    rows=[row for row in m.pair_rows() if 'election2020' not in row[1:]]
    # Validate every source before creating any links.
    for run_id,a,b in rows:
        source=reuse_root/'node_neighbors/lp'/run_id
        s=json.loads((source/'summary.json').read_text())
        if s['status']!='complete' or s['sources']!=[a,b]:raise ValueError(f'invalid reusable run {source}')
        if not (source/'best.pt').is_file():raise FileNotFoundError(source/'best.pt')
    parent=root/'node_neighbors/lp';parent.mkdir(parents=True,exist_ok=True)
    for run_id,_,_ in rows:
        source=(reuse_root/'node_neighbors/lp'/run_id).resolve();dest=parent/run_id
        try:dest.symlink_to(source,target_is_directory=True)
        except FileExistsError:
            if not dest.is_symlink() or dest.resolve()!=source:raise ValueError(f'reuse path conflict: {dest}')


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('phase');p.add_argument('--root',required=True)
    args,_=p.parse_known_args()
    m.SOURCES=tuple(SOURCE_ORDER)
    if args.phase=='train':reuse_completed(args.root)
    m.main()

if __name__=='__main__':main()
