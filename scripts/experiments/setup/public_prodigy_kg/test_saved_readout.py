import json
from pathlib import Path
import tempfile
import unittest

from .run_native import file_sha256
from .run_saved_readout import inspect_run


class SavedReadoutTests(unittest.TestCase):
    def test_source_receipts_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"fixture checkpoint, not deserialized")
            (root / "paired_episodes").mkdir()
            protocol = {"checkpoint_sha256": file_sha256(checkpoint),
                        "paired_mechanism": {"episode_count": 1}}
            (root / "protocol.json").write_text(json.dumps(protocol))
            status = root / "execution_status.json"
            status.write_text(json.dumps({"status": "complete"}))
            index = root / "paired_episodes/index.json"
            index.write_text(json.dumps({"records": [{"ordinal": 0, "file": "episode_00000.pt"}]}))
            self.assertEqual(len(inspect_run(root, checkpoint)[1]), 1)
            status.write_text(json.dumps({"status": "failed"}))
            with self.assertRaises(ValueError):
                inspect_run(root, checkpoint)
            status.write_text(json.dumps({"status": "complete"}))
            index.write_text(json.dumps({"records": [{"ordinal": 0, "file": "../other.pt"}]}))
            with self.assertRaises(ValueError):
                inspect_run(root, checkpoint)
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                inspect_run(root, checkpoint)


if __name__ == "__main__":
    unittest.main()
