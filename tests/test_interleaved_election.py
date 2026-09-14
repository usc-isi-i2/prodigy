import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from mixture_scaling import interleaved_election as e
from mixture_scaling.lattice import SOURCE_ORDER

class ElectionTests(unittest.TestCase):
    def test_extension_has_exactly_eight_new_pairs(self):
        with patch.object(e.m,'SOURCES',tuple(SOURCE_ORDER)):
            rows=e.m.pair_rows()
        self.assertEqual(len(rows),36)
        self.assertEqual(sum('election2020' in row[1:] for row in rows),8)

    def test_reuse_readonly_links_and_conflict_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old';new=Path(tmp)/'new';source=old/'node_neighbors/lp/pair';source.mkdir(parents=True)
            (old/'results').mkdir();(old/'results/COMPLETE.json').write_text(json.dumps(dict(status='complete',interleaved_models=28,evaluation_cells=224)))
            (source/'summary.json').write_text(json.dumps(dict(status='complete',sources=['A','B'])))
            (source/'best.pt').write_bytes(b'checkpoint')
            with patch.object(e.m,'pair_rows',return_value=[('pair','A','B')]):
                e.reuse_completed(new,old);e.reuse_completed(new,old)
                self.assertTrue((new/'node_neighbors/lp/pair').is_symlink())
                self.assertEqual((source/'best.pt').read_bytes(),b'checkpoint')
                with self.assertRaises(ValueError):e.reuse_completed(old,old)
                (new/'node_neighbors/lp/pair').unlink();(new/'node_neighbors/lp/pair').mkdir()
                with self.assertRaises(ValueError):e.reuse_completed(new,old)

if __name__=='__main__':unittest.main()
