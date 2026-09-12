"""Read-only verification in the production environment, including aligned attributes."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from nonzero_feature_views import SOURCES, CHUNK, EDGE_CHUNK, HISTORICAL, assert_equal, identity, verify_induced


def verify_attributes(source, view, ids):
    kept = torch.zeros(len(source['x']), dtype=torch.bool)
    kept[ids] = True
    def aligned(a, b):
        assert b.shape[0] == len(ids)
        for i in range(0, len(ids), CHUNK):
            assert_equal(a[ids[i:i + CHUNK]], b[i:i + CHUNK])
    def attrs(edges, a, b):
        offset = 0
        for start in range(0, edges.shape[1], EDGE_CHUNK):
            e = edges[:, start:start + EDGE_CHUNK]
            block = a[start:start + EDGE_CHUNK][kept[e[0]] & kept[e[1]]]
            assert_equal(block, b[offset:offset + len(block)])
            offset += len(block)
        assert offset == len(b)
    def walk(a, b):
        if hasattr(a, 'to_dict'):
            a, b = a.to_dict(), b.to_dict()
        for key, value in a.items():
            if key in HISTORICAL:
                assert_equal(value, view['source_graph_metadata'][key])
            elif key == 'edge_attr':
                attrs(a['edge_index'], value, b[key])
            elif key == 'edge_attr_views':
                for name, attr in value.items():
                    attrs(a['edge_index_views'][name], attr, b[key][name])
            elif 'edge_index' in key:
                continue  # independently checked by verify_induced
            elif key == 'u2i':
                assert b[key] == {u: i for i, u in enumerate(b['user_ids'])}
            elif isinstance(value, dict) or hasattr(value, 'to_dict'):
                walk(value, b[key])
            elif torch.is_tensor(value):
                aligned(value, b[key])
            elif key in ('handles', 'user_ids'):
                assert b[key] == [value[i] for i in ids.tolist()]
            elif key == 'num_nodes':
                assert b[key] == len(ids)
            else:
                assert_equal(value, b[key])
    walk(source, view)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, default=Path('docs/graph_catalog.json'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists()
    torch.set_num_threads(8)
    c = json.loads(args.catalog.read_text())
    entries = {e['dataset_key']: e for e in c['graphs']}
    results = []
    for key in SOURCES:
        start = time.monotonic()
        path = Path(c['data_root']) / entries[key]['relative_path']
        source = torch.load(path, map_location='cpu', weights_only=False)
        names = [key] + ([key + '_mini_500k'] if key in SOURCES[:2] else [])
        for name in names:
            d = args.root / name
            meta = json.loads((d / 'metadata.json').read_text())
            assert identity(path) == meta['source_identity']
            view = torch.load(d / 'graph.pt', map_location='cpu', weights_only=False)
            ids = view['original_node_ids']
            verify_induced(source, view, ids)
            verify_attributes(source, view, ids)
            if meta['sampling']:
                assert len(ids) == 500000
            record = {'dataset': name, 'verified': True, 'torch_version': torch.__version__,
                      'source_identity': identity(path), 'artifact_size_bytes': (d / 'graph.pt').stat().st_size,
                      'checks': ['exact features', 'all induced edge views', 'all aligned edge attributes',
                                 'labels and node targets', 'all masks', 'user mapping', 'embedded payload',
                                 'source unchanged'], 'seconds_from_source_start': time.monotonic() - start}
            results.append(record)
            print(json.dumps(record), flush=True)
            del view
            gc.collect()
        del source
        gc.collect()
    args.report.write_text(json.dumps(results, indent=2) + '\n')

if __name__ == '__main__':
    main()
